"""68 关：采样约束在投机下的语义（min_p / 惩罚 / min_tokens）与"未接入项"的显式拒绝。

验收点（需求 068 §3.1/§3.2/§3.6、§4）：

- temperature、top-k/top-p、min_p、三种惩罚、min_tokens 的**处理顺序**与"bonus 行 vs 候选行"
  的差别（顺序本身就是语义，不能按"常见采样公式"重排）；
- 逐行参数展开：第 j 个候选行看到的"假设历史"是"已提交 + 草稿前缀 [:j]"（不许重复应用）；
- 源码里本项目没有的字段（白名单 / bad words / logit_bias / 思考预算 / logprob_token_ids /
  prompt_logprobs）必须在**请求期**报错，不能静默忽略。
"""

import pytest
import torch

from spec68_helpers import (  # noqa: E402
    make_engine,
    prob_rows,
    sampling_metadata,
)
from minivllm.sample import Sampler  # noqa: E402
from minivllm.sample.rejection_sampler import RejectionSampler  # noqa: E402
from minivllm.testing.spec_metadata import make_metadata  # noqa: E402


def hand_softmax(row):
    import math

    top = max(row)
    exps = [math.exp(value - top) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


# ---------------------------------------------------------------------------
# 1) min_p：算式、位置（温度之后 / top-k 之前）、不改 argmax
# ---------------------------------------------------------------------------


def test_min_p_masks_below_threshold():
    """手算：`阈值 = max(softmax(logits)) × min_p`；低于阈值的全部 -inf。

    例：logits=[2,1,0]，softmax≈[0.665,0.245,0.090]，min_p=0.2 → 阈值 0.133
    → token 2（0.090）被屏蔽，token 0/1 保留。
    """
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    probs = hand_softmax([2.0, 1.0, 0.0])
    out = Sampler.apply_min_p(logits.clone(), torch.tensor([0.2]))
    assert out[0, 0].item() == pytest.approx(2.0)
    assert out[0, 1].item() == pytest.approx(1.0)
    assert out[0, 2].item() == float("-inf")
    assert probs[2] < probs[0] * 0.2               # 被屏蔽的那个确实低于阈值


def test_min_p_none_is_a_no_op():
    """整批没人用 min_p 时不动 logits（上游 `min_p_count == 0` 的早退分支）。"""
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    out = Sampler.apply_min_p(logits.clone(), None)
    assert out.tolist() == logits.tolist()


def test_min_p_is_applied_after_temperature():
    """min_p 的位置：**温度之后**（上游 `apply_temperature` → argmax 不变处理器 → top-k/top-p）。

    先除温度再算阈值与"先算阈值再除温度"结果不同：这里构造一份能让两种顺序给出不同掩码的
    logits，断言实现走的是上游那条顺序。
    """
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    temperature, min_p = 0.25, 0.001          # 低温把分布压尖 → 两种顺序的掩码不同
    after = hand_softmax((logits[0] / temperature).tolist())
    before = hand_softmax(logits[0].tolist())
    assert after[2] < after[0] * min_p                 # 温度之后：token 2 被屏蔽
    assert after[1] >= after[0] * min_p                #           token 1 保留
    assert before[2] >= before[0] * min_p              # 温度之前：token 2 不会被屏蔽

    sm = sampling_metadata([temperature], [[]], min_p=[min_p], top_k=None, top_p=None,
                           logprobs_mode="processed_logits", max_num_logprobs=3)
    out = Sampler("processed_logits").forward(logits.clone(), sm)
    # processed_logits 交付的就是筛选后的 logits：走的是"温度之后"那条顺序
    tensors = out.logprobs_tensors
    columns = tensors.logprob_token_ids[0].tolist()
    row = tensors.logprobs[0]
    assert row[columns.index(0)].item() == pytest.approx(3.0 / temperature)
    assert row[columns.index(1)].item() == pytest.approx(2.0 / temperature)
    assert row[columns.index(2)].item() == float("-inf")


def test_min_p_never_changes_greedy_argmax():
    """min_p 是"argmax 不变"的约束（上游 `MinPLogitsProcessor.is_argmax_invariant`）。

    贪心行根本不走 min_p；即使走了，最大值也永远高于"最大概率 × min_p"，不会被屏蔽。
    """
    logits = torch.tensor([[5.0, 0.0, -3.0], [1.0, 1.0, 1.0]])
    sm = sampling_metadata([0.0, 0.0], [[], []], min_p=[0.9, 0.9],
                           logprobs_mode="raw_logprobs", max_num_logprobs=1)
    out = Sampler("raw_logprobs").forward(logits.clone(), sm)
    assert out.sampled_token_ids.flatten().tolist() == logits.argmax(dim=-1).tolist()


# ---------------------------------------------------------------------------
# 2) 投机路径的逐行参数：假设历史、不重复应用
# ---------------------------------------------------------------------------


def test_spec_rows_get_assumed_history_once(cuda_device):
    """候选行 j 的惩罚历史 = 已提交历史 + 草稿前缀 `[:j]`；每一行**只施加一次**。

    走真投机路径（`RejectionSampler.apply_logits_processors`）：它按请求把惩罚参数用
    Triton 内核 `expand_kernel` 展开到逐行，所以这条用例需要 CUDA（CPU 上的算法语义由
    `minivllm/testing/torch_rejection_sampler.py` 覆盖）。
    

    手算（frequency penalty，按次数减）：
        已提交历史 = [kv]，草稿 = [kv, other]
        第 0 行历史 = [kv]        → kv 计数 1 → logit -= 1.0×0.5
        第 1 行历史 = [kv, kv]    → kv 计数 2 → logit -= 1.0×2×0.5
    """
    kv, other = 2, 3
    logits = prob_rows([[2.0, 1.0, 2.0, 1.0], [2.0, 1.0, 2.0, 1.0]], device=cuda_device)
    meta = make_metadata([[kv, other]], device=cuda_device)
    sm = sampling_metadata(
        [0.0], [[kv, other]], device=cuda_device, max_num_logprobs=None,
        no_penalties=False, output_token_ids=[[kv]], prompt_token_ids=[[]],
        frequency_penalties=[1.0], presence_penalties=[0.0], repetition_penalties=[1.0])
    rs = RejectionSampler(Sampler("raw_logprobs"))
    out = rs.apply_logits_processors(logits.clone(), sm, meta)

    # prob_rows 把概率 p 变成 log(p)：kv 的**原始** logit 是 ln(2)≈0.693、other 是 ln(1)=0
    import math

    # 第 0 行：历史 [kv] → kv 计数 1 → 减 1.0×1
    assert out[0, kv].item() == pytest.approx(math.log(2.0) - 1.0, abs=1e-6)
    # 第 1 行：历史 [kv, kv] → 计数 2 → 减 1.0×2（**不是** 1.0 的重复叠加成 3 份）
    assert out[1, kv].item() == pytest.approx(math.log(2.0) - 2.0, abs=1e-6)
    # 草稿第 2 枚（other）只在**更后面**的行的前缀里；前缀 [:1] 只含 kv → 第 1 行不含 other
    assert out[1, other].item() == pytest.approx(math.log(1.0), abs=1e-6)


def test_bonus_row_history_includes_all_drafts():
    """bonus 行（能走到它就说明草稿全被接受）历史 = 已提交 + **全部**草稿。

    同一个 token 在草稿里出现两次时，频率惩罚必须减 2×0.5（少算会让 bonus 的分布变样，
    而 bonus 是最终交付的 token 之一——静默错）。
    """
    kv = 2
    logits = prob_rows([[2.0, 1.0, 2.0, 1.0]])
    sm = sampling_metadata(
        [0.0], [[kv, kv]], no_penalties=False, output_token_ids=[[]],
        prompt_token_ids=[[]], frequency_penalties=[1.0], presence_penalties=[0.0],
        repetition_penalties=[1.0])
    out = Sampler("raw_logprobs").forward(logits.clone(), sm, predict_bonus_token=True)
    # 直接比较"施加惩罚后的 logits 是否等于手算值"：用 processed_logits 模式再看一次
    sm_processed = sampling_metadata(
        [0.0], [[kv, kv]], no_penalties=False, output_token_ids=[[]],
        prompt_token_ids=[[]], frequency_penalties=[1.0], presence_penalties=[0.0],
        repetition_penalties=[1.0], logprobs_mode="processed_logits", max_num_logprobs=4)
    processed = Sampler("processed_logits").forward(
        logits.clone(), sm_processed, predict_bonus_token=True)
    import math

    tensors = processed.logprobs_tensors
    column = tensors.logprob_token_ids[0].tolist().index(kv)   # 列序是 [选中, top1, ...]
    # processed_logits 交付的是 logits：kv 原始 logit ln(2)，草稿里出现 2 次 → 减 1.0×2
    assert tensors.logprobs[0, column].item() == pytest.approx(math.log(2.0) - 2.0, abs=1e-6)
    assert out.sampled_token_ids.flatten().tolist() == [0]


def test_min_tokens_masks_only_the_first_rows_of_a_draft_block():
    """`min_tokens` 在投机下屏蔽的是"还没生成够的那些候选行"（上游 `apply_with_spec_decode`）。

    历史长度 = 已提交 + j；要屏蔽停止 token 的行是前 `clamp(min_tokens - len(out), 0, K)` 行。
    """
    stop = 1
    logits = torch.full((3, 4), 0.5)          # K=3 个候选行，停止 token 是 argmax
    logits[:, stop] = 5.0
    sm = sampling_metadata([0.0], [[2, 2, 2]], min_tokens=[2], stop_token_ids=[[stop]],
                           output_token_ids=[[]])   # 已生成 0 个 → 前 2 行屏蔽
    out = Sampler("raw_logprobs").apply_min_tokens_for_spec_decode(
        logits.clone(), sm, [3])
    assert out[0, stop].item() == float("-inf")
    assert out[1, stop].item() == float("-inf")
    assert out[2, stop].item() == pytest.approx(5.0)      # 第 3 行历史长度 2 → 够 min_tokens


# ---------------------------------------------------------------------------
# 3) 未接入的能力：请求期必须报错（不许静默忽略）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,match", [
    ({"allowed_token_ids": [1, 2]}, "allowed_token_ids"),
    ({"bad_words": ["bad"]}, "bad_words"),
    ({"logit_bias": {1: 0.5}}, "logit_bias"),
    ({"thinking_token_budget": 16}, "thinking_token_budget"),
    ({"logprob_token_ids": [1, 2]}, "logprob_token_ids"),
    ({"prompt_logprobs": 2}, "prompt_logprobs"),
])
def test_unwired_sampling_params_are_rejected(kwargs, match):
    """这三态矩阵里属「本项目尚未接入」的字段：收下参数就报错（068 §3.6）。"""
    from minivllm import SamplingParams

    with pytest.raises(NotImplementedError, match=match):
        SamplingParams(**kwargs)


def test_invalid_logprobs_and_min_p_are_rejected():
    from minivllm import SamplingParams

    with pytest.raises(ValueError, match="logprobs"):
        SamplingParams(logprobs=0)
    with pytest.raises(ValueError, match="min_p"):
        SamplingParams(min_p=1.5)


def test_logprobs_above_max_logprobs_is_rejected_at_request_time(tiny_dir, hf_config):
    """要的 logprobs 超过 `ModelConfig.max_logprobs` → 提交请求时就报错（上游同款）。

    静默少给几个最坏：下游以为拿到的是"完整的前 k 名"，实际上名次在第 k 位就断了。
    """
    from minivllm import SamplingParams

    engine, _core, _runner = make_engine(model_dir=tiny_dir, hf_config=hf_config,
                                        device="cpu", max_logprobs=3)
    try:
        with pytest.raises(ValueError, match="max_logprobs"):
            engine.add_request("r1", [1, 2, 3], SamplingParams(max_tokens=2, logprobs=5))
        # 边界内可以正常提交
        engine.add_request("r2", [1, 2, 3], SamplingParams(max_tokens=2, logprobs=3))
    finally:
        engine.shutdown()


def test_unwired_backend_is_rejected(structured_config, structured_tokenizer):
    """`structured_outputs.backend` 只接 xgrammar / auto；别的值在**建 grammar 时**报错。"""
    from minivllm import SamplingParams, StructuredOutputsParams
    from minivllm.config import StructuredOutputsConfig
    from minivllm.structured_output import StructuredOutputManager

    for backend in ("guidance", "outlines", "lm-format-enforcer"):
        manager = StructuredOutputManager(_config_with_backend(structured_config, backend))
        params = SamplingParams(max_tokens=2,
                                structured_outputs=StructuredOutputsParams(json_object=True))
        request = _fake_request(params)
        with pytest.raises(NotImplementedError, match="backend"):
            manager.grammar_init(request)


def _config_with_backend(model_config, backend):
    from minivllm.config import StructuredOutputsConfig, SchedulerConfig, VllmConfig

    return VllmConfig(model_config=model_config,
                      scheduler_config=SchedulerConfig(max_num_seqs=2,
                                                       max_num_batched_tokens=8),
                      structured_outputs_config=StructuredOutputsConfig(backend=backend))


def _fake_request(sampling_params, request_id="r1"):
    from minivllm.request import Request

    return Request(request_id, [1, 2, 3], sampling_params, arrival_time=1.0)


def test_min_p_only_applies_to_the_bonus_row(cuda_device):
    """需求 068 §3.2 的"bonus 行 vs 候选行不同之处"：`min_p` **只作用在 bonus 行**。

    上游 `min_p` 是"argmax 不变"的 logits processor（`MinPLogitsProcessor`，在
    `Sampler.sample()` 里施加）。投机路径的**候选验证行不经过 `Sampler.sample()`**——
    它只走 `RejectionSampler.apply_logits_processors`（惩罚/白名单/bad words/min_tokens/思考预算）
    与 `apply_sampling_constraints`（温度/top-k/top-p），所以候选行上**没有** min_p 掩码。
    本仓库照抄这条行为（上游没有的部分我们不自创），这条用例把它钉住：

        · 候选行：`min_p=1.0`（只该留概率最大的那个）之后仍然**一个 -inf 都没有**
        · bonus 行：同一个 min_p 生效 → 除 argmax 外全是 -inf

    后果（已量到，见 `vllm_bugs/UPSTREAM_SUSPECTED_BUGS.md` 的 BUG-6）：投机下被接受的草稿
    可以是 `min_p` 本该屏蔽的 token，于是"开投机"与"不开投机"的输出分布对 `min_p` 的承诺不一致。
    """
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], device=cuda_device)
    sm = sampling_metadata([1.0], [[]], device=cuda_device, min_p=[1.0],
                           logprobs_mode="processed_logits", max_num_logprobs=4)

    # ---- 候选行：走 apply_sampling_constraints（温度 + top-k/top-p），没有 min_p ----
    from minivllm.sample.rejection_sampler import apply_sampling_constraints
    # cu_num_draft_tokens=[1] = 这条请求有 1 个候选行（K=0 的请求不占候选行，见 59 关）
    candidate = apply_sampling_constraints(
        logits.clone(), torch.tensor([1], dtype=torch.int32, device=cuda_device), sm)
    assert torch.equal(candidate, logits), (
        f"候选行被改动了（温度=1.0、top-k/top-p 都没开，唯一可能的改动者就是 min_p）："
        f"{candidate.tolist()}")

    # ---- bonus 行：走 Sampler.sample → min_p 生效 ----
    bonus = Sampler("processed_logits").forward(logits.clone(), sm)
    columns = bonus.logprobs_tensors.logprob_token_ids[0].tolist()
    values = bonus.logprobs_tensors.logprobs[0].tolist()
    assert values[columns.index(0)] == pytest.approx(2.0)      # argmax（token 0）留下
    for column, token_id in enumerate(columns):
        if token_id != 0:
            assert values[column] == float("-inf"), (columns, values)
