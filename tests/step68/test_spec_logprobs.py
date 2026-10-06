"""68 关：logprobs 的逐值对齐（四种模式 / 投机行索引 / 截断 / 未接入项）。

验收点（需求 068 §3.5、§4）：

- raw / processed 两套 logprobs 小词表**手算**；
- 接受候选、恢复 token、bonus **各自索引正确**（"第 j 个位置读第 j 行"）；
- 被拒绝的候选 / 被截断的尾部**不漏出**；
- 与上游同层对象逐值差分（`Sampler.forward` 与 `RejectionSampler._get_logprobs_tensors`）。
"""

import math

import pytest
import torch

from spec68_helpers import (  # noqa: E402
    ours_sampler_output,
    prob_rows,
    sampling_metadata,
    upstream_sampler_output,
    upstream_sampling_metadata,
    upstream_spec_metadata,
)
from minivllm.outputs import LogprobsTensors  # noqa: E402
from minivllm.sample import Sampler  # noqa: E402
from minivllm.sample.rejection_sampler import RejectionSampler  # noqa: E402
from minivllm.testing.spec_metadata import make_metadata  # noqa: E402

MODES = ("raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits")


def hand_log_softmax(row):
    total = sum(math.exp(value) for value in row)
    return [value - math.log(total) for value in row]


# ---------------------------------------------------------------------------
# 1) 非投机：四种模式的逐值手算 + 与上游差分
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_greedy_logprobs_match_hand_computation(mode):
    """贪心行：手算 `log_softmax`（raw 模式）/ 处理过的 logits（processed 模式）。"""
    # 词表 4 的 logits；没有任何惩罚/温度（贪心行不施加温度与 top-k/top-p）
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, 0.5]])
    sm = sampling_metadata([0.0, 0.0], [[], []], max_num_logprobs=2, logprobs_mode=mode)
    out = ours_sampler_output(logits, sm)
    tensors = out.logprobs_tensors

    sampled = out.sampled_token_ids.flatten().tolist()
    assert sampled == [0, 1]
    # 第 0 列 = 采到的 token；后面是 top-k
    assert tensors.logprob_token_ids[0].tolist() == [0, 0, 1]
    # 第 1 行 logits=[0,2,1,0.5]：top1=1、top2=2（采到的 1 也在 top-k 里，只留一份）
    assert tensors.logprob_token_ids[1].tolist() == [1, 1, 2]
    assert tensors.selected_token_ranks.tolist() == [1, 1]

    # *logprobs 模式交付 log_softmax 之后的值；*logits 模式交付 logits 本身
    expected = (hand_log_softmax(logits[0].tolist()) if mode.endswith("logprobs")
                else logits[0].tolist())
    assert tensors.logprobs[0][0].item() == pytest.approx(expected[0], abs=1e-6)
    assert tensors.logprobs[0][1].item() == pytest.approx(expected[0], abs=1e-6)
    assert tensors.logprobs[0][2].item() == pytest.approx(expected[1], abs=1e-6)


def test_processed_mode_differs_from_raw_when_penalties_apply():
    """`processed_*` 与 `raw_*` 的差别必须来自**真的被改过**：这里用三种惩罚。

    三条语义（逐条对着上游 `apply_all_penalties` 的算式，别按"常见写法"猜）：

        repetition_penalty   token **出现过就按一次**缩放（与出现几次无关）：正 logit 除以 p
        presence_penalty     token 出现过就减一次 presence
        frequency_penalty    按**出现次数**减：`logit -= frequency × count`

    只有频率惩罚是"次数相关"的——它就是"历史里算不算上草稿"会改变数字的那一项
    （见 docs/step68_alignment.md 的痛点算例）。
    """
    token_kv = 2
    logits = torch.tensor([[3.0, 1.0, 3.0, 0.0]])
    output_token_ids = [[token_kv, token_kv]]        # 出现两次

    def spec_metadata(mode):
        return sampling_metadata(
            [0.0], [[]], max_num_logprobs=3, logprobs_mode=mode,
            output_token_ids=output_token_ids, prompt_token_ids=[[]], no_penalties=False,
            repetition_penalties=[1.2], presence_penalties=[0.0],
            frequency_penalties=[0.5])

    raw = ours_sampler_output(logits, spec_metadata("raw_logprobs"))
    processed = ours_sampler_output(logits, spec_metadata("processed_logprobs"))

    raw_kv = raw.logprobs_tensors.logprobs[0, raw.logprobs_tensors.logprob_token_ids[0]
                                           .tolist().index(token_kv)]
    # raw：没有惩罚 → 与 log_softmax(原始 logits) 一致
    assert raw_kv.item() == pytest.approx(hand_log_softmax([3.0, 1.0, 3.0, 0.0])[token_kv],
                                          abs=1e-6)
    # processed：惩罚之后 kv 的概率被压低 → logprob 一定比 raw 小
    processed_kv = processed.logprobs_tensors.logprobs[
        0, processed.logprobs_tensors.logprob_token_ids[0].tolist().index(token_kv)]
    assert processed_kv.item() < raw_kv.item() - 0.05
    # 手算：3/1.2（重复惩罚只算一次） - 0.5×2（频率惩罚按次数）
    penalized = [3.0, 1.0, 3.0 / 1.2 - 0.5 * 2, 0.0]
    assert processed_kv.item() == pytest.approx(hand_log_softmax(penalized)[token_kv],
                                                abs=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_sampler_logprobs_match_upstream(mode):
    """非投机路径与上游 `Sampler.forward` 逐值差分（同一份 logits + 同一份元数据）。"""
    logits = torch.tensor([[1.5, 0.2, -0.7, 2.2], [0.1, 0.1, 0.1, 0.1]])
    # 随机行（第 1 行）两边给**同一个种子**的 generator：算法相同 → 抽到的 token 也应相同
    # （总纲 §5：比较采样系统时比分布，不强求不同实现同 seed 逐 token 相等；这里两边是同一套
    # 指数竞赛，所以顺带把 token 也比上，能多抓一类"分布不同"的错）。
    ours_gen = {1: torch.Generator().manual_seed(7)}
    up_gen = {1: torch.Generator().manual_seed(7)}
    common = dict(output_token_ids=[[1], []], prompt_token_ids=[[0], [0]],
                  no_penalties=False, repetition_penalties=[1.1, 1.0],
                  presence_penalties=[0.0, 0.0], frequency_penalties=[0.0, 0.2])
    ours_sm = sampling_metadata([0.0, 0.7], [[], []], max_num_logprobs=2,
                                logprobs_mode=mode, generators=ours_gen, **common)
    up_sm = upstream_sampling_metadata([0.0, 0.7], [[], []], max_num_logprobs=2,
                                       generators=up_gen, **common)
    ours = ours_sampler_output(logits, ours_sm, mode=mode)
    up = upstream_sampler_output(logits, up_sm, mode=mode)

    assert ours.sampled_token_ids.tolist() == up.sampled_token_ids.tolist()
    assert (ours.logprobs_tensors.logprob_token_ids.tolist()
            == up.logprobs_tensors.logprob_token_ids.tolist())
    assert (ours.logprobs_tensors.selected_token_ranks.tolist()
            == up.logprobs_tensors.selected_token_ranks.tolist())
    assert torch.allclose(ours.logprobs_tensors.logprobs,
                          up.logprobs_tensors.logprobs, atol=1e-6)


def test_rank_counts_ties_as_equal():
    """名次 = "大于等于选中 logprob 的个数"（并列算同一档，上游同款）。"""
    logits = torch.tensor([[1.0, 1.0, 0.0, -1.0]])
    sm = sampling_metadata([0.0], [[]], max_num_logprobs=1, logprobs_mode="raw_logprobs")
    out = ours_sampler_output(logits, sm)
    # 选中的是 token 0（并列里取下标小的），token 1 与它同分 → 名次 2
    assert out.sampled_token_ids.flatten().tolist() == [0]
    assert out.logprobs_tensors.selected_token_ranks.tolist() == [2]


def test_full_vocab_mode_shape_matches_upstream():
    """`logprobs=-1`（全词表）：采样器交出整份分布、不排名次（上游同款形状）。

    注意这条**不是**"用户能拿到全词表"：输出处理阶段按 top-k 那一列解释（上游同款 zip），
    实测最终交付为空 —— 见 `test_full_vocab_mode_is_truncated_at_output_processing`。
    """
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    sm = sampling_metadata([0.0], [[]], max_num_logprobs=-1, logprobs_mode="raw_logprobs")
    ours = ours_sampler_output(logits, sm)
    up = upstream_sampler_output(logits, upstream_sampling_metadata(
        [0.0], [[]], max_num_logprobs=-1), mode="raw_logprobs")
    assert ours.logprobs_tensors.logprob_token_ids.shape == (0,)
    assert ours.logprobs_tensors.selected_token_ranks.shape == (0,)
    assert ours.logprobs_tensors.logprobs.shape == up.logprobs_tensors.logprobs.shape
    assert torch.allclose(ours.logprobs_tensors.logprobs,
                          up.logprobs_tensors.logprobs, atol=1e-6)
    # 全词表那一份就是 log_softmax（raw_logprobs 模式）
    assert torch.allclose(ours.logprobs_tensors.logprobs[0],
                          torch.tensor(hand_log_softmax([2.0, 1.0, 0.0, -1.0])), atol=1e-6)


# ---------------------------------------------------------------------------
# 2) 投机：行索引（接受 / 恢复 / bonus）+ 与上游差分
# ---------------------------------------------------------------------------


def _spec_setup(mode, drafts=(0, 1, 2), rows=None):
    """造一份"每请求 K 行候选 + 1 行 bonus"的紧凑 logits（行序：候选在前、bonus 在后）。

    默认行是**普通 logits**（不是归一化概率）：第 j 行的 argmax 是 token j，
    所以"接受第 j 个候选"这种场景下的期望值就是 `log_softmax(第 j 行)[j]`，
    不会出现"两边都是 -inf 所以比相等"的空断言。
    """
    meta = make_metadata([list(drafts)], device="cpu")
    if rows is None:
        rows = [[2.0, 1.0, 0.5, 0.2], [0.2, 2.0, 1.0, 0.5],
                [0.5, 0.2, 2.0, 1.0], [1.0, 0.5, 0.2, 2.0]]
    logits = torch.tensor(rows, dtype=torch.float32)
    sampler = Sampler(mode)
    rs = RejectionSampler(sampler)
    return meta, logits, sampler, rs


def _bonus_logits_for(sampler, meta, logits, mode, sm=None):
    """bonus 采样器交回的那一份（生产路径里由 `RejectionSampler.forward` 传进来）。

    `sm` 给定时用它（这样惩罚/温度等设置与生产路径一致）；不给就用一份"无惩罚"的默认元数据。
    """
    from dataclasses import replace

    if sm is None:
        sm = sampling_metadata([0.0], [[3, 1, 2]])
    sm = replace(sm, max_num_logprobs=-1)
    out = sampler.forward(logits[meta.bonus_logits_indices], sm,
                          predict_bonus_token=True,
                          logprobs_mode_override=("processed_logits"
                                                  if mode.startswith("processed")
                                                  else "raw_logits"))
    return out.logprobs_tensors.logprobs


@pytest.mark.parametrize("mode", MODES)
def test_spec_logprobs_match_upstream(mode):
    """投机路径与上游 `_get_logprobs_tensors` 逐值差分（含被拒候选位与 bonus 位）。"""
    meta, logits, sampler, rs = _spec_setup(mode)
    up_meta = upstream_spec_metadata([[3, 1, 2]], device="cpu")
    from vllm.v1.sample.rejection_sampler import RejectionSampler as UpstreamRejection

    target = logits[meta.target_logits_indices]
    bonus = _bonus_logits_for(sampler, meta, logits, mode)
    sampled = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)   # 三个候选都接受 + bonus

    ours = rs._get_logprobs_tensors(2, meta, logits, target, bonus, sampled)
    up = UpstreamRejection(
        sampler=__import__("vllm.v1.sample.sampler", fromlist=["Sampler"]).Sampler(
            logprobs_mode=mode),
    )._get_logprobs_tensors(2, up_meta, logits, target, bonus, sampled)

    assert ours.logprob_token_ids.tolist() == up.logprob_token_ids.tolist()
    assert ours.selected_token_ranks.tolist() == up.selected_token_ranks.tolist()
    assert torch.allclose(ours.logprobs, up.logprobs, atol=1e-6)


def test_accepted_draft_reads_its_own_row():
    """接受第 j 个候选 → 它的 logprob 来自**第 j 行**的分布（行与位置一一对应）。"""
    meta, logits, sampler, rs = _spec_setup("raw_logprobs")
    target = logits[meta.target_logits_indices]
    bonus = _bonus_logits_for(sampler, meta, logits, "raw_logprobs")
    sampled = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    got = rs._get_logprobs_tensors(1, meta, logits, target, bonus, sampled)

    expected = [hand_log_softmax(logits[row].tolist())[token]
                for row, token in enumerate([0, 1, 2, 3])]
    assert got.logprob_token_ids[:, 0].tolist() == [0, 1, 2, 3]
    for slot, value in enumerate(expected):
        assert got.logprobs[slot, 0].item() == pytest.approx(value, abs=1e-5)


def test_recovered_token_at_slot_a_reads_row_a():
    """第 a 个位置被拒 → 恢复 token 的 logprob 也来自**第 a 行**（同一个位置的 p）。

    这里 p 是"恢复 token 独占"的行，所以只要读对了行，logprob 就接近 0（概率接近 1）；
    读错行（比如读 bonus 行）会得到明显不同的数。
    """
    meta, logits, sampler, rs = _spec_setup(
        "raw_logprobs",
        rows=[[0.0, 0.0, 0.0, 5.0],       # 第 0 行：argmax = token 3（草稿 3 被接受）
              [0.0, 5.0, 0.0, 0.0],       # 第 1 行：argmax = token 1（恢复 token）
              [0.0, 0.0, 0.0, 5.0],
              [5.0, 0.0, 0.0, 0.0]])      # bonus
    target = logits[meta.target_logits_indices]
    bonus = _bonus_logits_for(sampler, meta, logits, "raw_logprobs")
    # 第 0 个候选接受（token 3），第 1 个位置被拒 → 恢复 token 1，后面全是 -1
    sampled = torch.tensor([[3, 1, -1, -1]], dtype=torch.int32)
    got = rs._get_logprobs_tensors(2, meta, logits, target, bonus, sampled)

    assert got.logprob_token_ids[:, 0].tolist() == [3, 1, 0, 0]   # -1 被换成 0 参与计算
    assert got.logprobs[1, 0].item() == pytest.approx(
        hand_log_softmax(logits[1].tolist())[1], abs=1e-5)
    # 反证：如果错读 bonus 行（第 3 行），数字会完全不同
    assert abs(got.logprobs[1, 0].item()
               - hand_log_softmax(logits[3].tolist())[1]) > 1.0


def test_bonus_position_reads_bonus_row():
    """全部接受 → 最后那个位置读 bonus 行。"""
    meta, logits, sampler, rs = _spec_setup("raw_logprobs")
    target = logits[meta.target_logits_indices]
    bonus = _bonus_logits_for(sampler, meta, logits, "raw_logprobs")
    sampled = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    got = rs._get_logprobs_tensors(1, meta, logits, target, bonus, sampled)
    assert got.logprobs[3, 0].item() == pytest.approx(
        hand_log_softmax(logits[3].tolist())[3], abs=1e-5)


def test_processed_logprobs_bonus_row_is_numerically_correct():
    """`processed_logprobs` 模式下 bonus 位的**交付值 == 手算的 processed logprob**。

    2026-10-06 独立复核的结论（原先我们记成"上游疑似 bug"，是假阳性）：上游确实会"双
    `log_softmax`"——bonus 采样器在 `processed_logprobs` 模式下交回的本就是 logprobs，
    `_get_logprobs_tensors` 又对它做一次——但 **`log_softmax` 幂等**（对已归一化的向量再取一次
    log_softmax 得到它自己），所以数字没错。

    这条用例用**生产路径的口径**钉住它：惩罚由 bonus 采样器自己施加（`predict_bonus_token=True`，
    历史 = 已提交 + 全部草稿），再用**手算的 processed 分布**做期望值。
    """
    rows = [[2.0, 1.0, 0.5, 0.2], [0.2, 2.0, 1.0, 0.5], [0.5, 0.2, 2.0, 1.0],
            [1.0, 0.5, 0.2, 2.0]]
    # K=2（草稿 [0,1]）、bonus 行是第 2 行 → 词表里的 token 2 不在历史里，惩罚**不是均匀的**
    meta, logits, sampler, rs = _spec_setup("processed_logprobs", drafts=(0, 1), rows=rows)
    # bonus 行的历史 = 已提交 [3] + 草稿 [0,1]；frequency_penalty=1.0 → token 0/1/3 各减 1.0、token 2 不动
    sm = sampling_metadata([0.0], [[0, 1]], max_num_logprobs=None, no_penalties=False,
                           output_token_ids=[[3]], prompt_token_ids=[[]],
                           frequency_penalties=[1.0], presence_penalties=[0.0],
                           repetition_penalties=[1.0])
    bonus = _bonus_logits_for(sampler, meta, logits, "processed_logprobs", sm=sm)

    sampled = torch.tensor([[0, 1, 3]], dtype=torch.int32)
    got = rs._get_logprobs_tensors(1, meta, logits, logits[meta.target_logits_indices],
                                   bonus, sampled)

    bonus_row = int(meta.bonus_logits_indices[0])   # 行号由 metadata 给（别写死下标）
    penalized_bonus = logits[bonus_row].tolist()
    for token_id in (0, 1, 3):                     # 历史 = [3, 0, 1]，各出现 1 次
        penalized_bonus[token_id] -= 1.0
    hand_processed = hand_log_softmax(penalized_bonus)[3]
    hand_raw = hand_log_softmax(logits[bonus_row].tolist())[3]
    assert abs(hand_processed - hand_raw) > 0.1, (hand_processed, hand_raw)   # 两份确实不同
    assert got.logprobs[2, 0].item() == pytest.approx(hand_processed, abs=1e-5)
    # 反证：如果交付的真是 raw 那一份（我们原先的怀疑），这里会差 0.1 以上
    assert abs(got.logprobs[2, 0].item() - hand_raw) > 0.1


def test_log_softmax_is_idempotent():
    """复核结论的机理：`log_softmax(log_softmax(x)) == log_softmax(x)`（数值上 `max|Δ|=0`）。

    这就是"上游双 log_softmax 但没有数值后果"的原因；把这条单独钉住，免得以后有人
    （包括我们自己）再把它当成 bug 重新报一遍。
    """
    x = torch.tensor([[2.0, 1.0, 0.5, 0.2], [0.1, 0.1, 0.1, 0.1]])
    once = x.log_softmax(dim=-1, dtype=torch.float32)
    twice = once.log_softmax(dim=-1, dtype=torch.float32)
    assert (once - twice).abs().max().item() < 1e-6, (once - twice).abs().max().item()


def test_spec_full_vocab_mode_is_rejected_until_aligned():
    """投机 + `logprobs=-1`：上游在 `torch.topk(k=-1)` 处运行期报错，本项目提前拒绝。

    上游证据（差分测试实测）：`torch.topk(x, -1)` → `RuntimeError: selected index k out of
    range`。两边都是"不支持"，本项目把失败点提前并说清原因。
    """
    meta, logits, sampler, rs = _spec_setup("raw_logprobs")
    target = logits[meta.target_logits_indices]
    bonus = _bonus_logits_for(sampler, meta, logits, "raw_logprobs")
    sampled = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="全词表"):
        rs._get_logprobs_tensors(-1, meta, logits, target, bonus, sampled)
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        torch.topk(torch.randn(1, 4), -1, dim=-1)


# ---------------------------------------------------------------------------
# 3) 截断：被拒候选 / 停止 token 的尾巴都不能漏出
# ---------------------------------------------------------------------------


def test_parse_output_filters_tokens_and_logprobs_with_same_mask():
    """`parse_output` 用**同一张** valid_mask 同时裁 token 与 logprobs。"""
    meta, logits, sampler, rs = _spec_setup("raw_logprobs")
    target = logits[meta.target_logits_indices]
    bonus = _bonus_logits_for(sampler, meta, logits, "raw_logprobs")
    sampled = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)   # 第 2 位起被拒
    tensors = rs._get_logprobs_tensors(1, meta, logits, target, bonus, sampled)

    rows, lists = RejectionSampler.parse_output(sampled, vocab_size=4,
                                                logprobs_tensors=tensors)
    assert rows == [[0, 1]]
    # 只有 2 个位置活着；`cu_num_generated_tokens` 让"第 0 条请求从第 0 行开始、共 2 行"
    assert lists.cu_num_generated_tokens == [0, 2]
    assert lists.logprob_token_ids.shape == (2, 2)
    assert lists.logprob_token_ids[:, 0].tolist() == [0, 1]
    # 被拒的候选位（-1 换成 0 参与过计算）**没有**出现在交付里
    assert lists.logprob_token_ids[:, 0].tolist() != [0, 0, 1, 0]


def test_parse_output_multiple_requests_get_their_own_offsets():
    """两条请求、接受数不同（K=3 与 K=0）：行偏移必须各归各的。"""
    meta = make_metadata([[1, 1, 1], []], device="cpu")
    # 紧凑行序：请求 0 的 3 个候选（行 0~2）+ bonus（行 3）、请求 1 的 bonus（行 4）
    logits = prob_rows([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [1.0, 3.0], [3.0, 0.0]])
    rs = RejectionSampler(Sampler("raw_logprobs"))
    target = logits[meta.target_logits_indices]
    bonus = logits[meta.bonus_logits_indices]
    # 请求 0：第 0 个候选接受、第 1 个被拒（恢复 token 1）→ 交付 2 个位置
    # 请求 1（K=0）：只有 bonus 一个位置
    sampled = torch.tensor([[1, 1, -1, -1], [0, -1, -1, -1]], dtype=torch.int32)
    tensors = rs._get_logprobs_tensors(1, meta, logits, target, bonus, sampled)
    rows, lists = RejectionSampler.parse_output(sampled, vocab_size=2,
                                               logprobs_tensors=tensors)
    assert rows == [[1, 1], [0]]
    assert lists.cu_num_generated_tokens == [0, 2, 3]
    # 请求 1 的位置从第 2 行开始（请求 0 占了 2 行），读的是它自己那一行（紧凑第 4 行）
    assert lists.logprob_token_ids[2, 0].item() == 0
    assert lists.logprobs[2, 0].item() == pytest.approx(
        hand_log_softmax(logits[4].tolist())[0], abs=1e-5)


def test_scheduler_truncates_logprobs_with_stopped_tail():
    """停止 token 把本轮候选截断时，logprobs 同步截断（Scheduler 按提交后的数量切）。

    用真 Scheduler + 手工 `ModelRunnerOutput`：请求 `max_tokens=1`，这一轮"采到" 3 个 token，
    但第 1 个就触发停止 → 只提交 1 个 token、只交付 1 个位置的 logprobs。
    """
    import numpy as np

    from minivllm import SamplingParams
    from minivllm.core.kv_cache_manager import KVCacheManager
    from minivllm.core.sched.scheduler import Scheduler
    from minivllm.config import CacheConfig, SchedulerConfig
    from minivllm.outputs import LogprobsLists, ModelRunnerOutput
    from minivllm.request import Request

    max_model_len = 32
    manager = KVCacheManager(CacheConfig(block_size=4, num_gpu_blocks=8),
                             max_model_len=max_model_len)
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=16),
                          manager, max_model_len=max_model_len)
    params = SamplingParams(max_tokens=1, min_tokens=0, eos_token_id=7, logprobs=2)
    request = Request("r1", [1, 2, 3], params, arrival_time=1.0)
    scheduler.add_request(request)
    packet = scheduler.schedule()
    logprobs = LogprobsLists(
        np.array([[1, 1, 2], [7, 7, 9], [5, 5, 6]], dtype=np.int32),
        np.array([[-0.1, -0.1, -2.0], [-0.2, -0.2, -3.0], [-0.3, -0.3, -4.0]],
                 dtype=np.float32),
        np.array([1, 1, 2], dtype=np.int32), [0, 3])
    output = ModelRunnerOutput(req_ids=["r1"], req_id_to_index={"r1": 0},
                               sampled_token_ids=[[1, 7, 5]],   # 第 2 个是 eos
                               logprobs=logprobs)
    result = scheduler.update_from_output(packet, output)
    core_output = result.outputs[0]
    assert core_output.new_token_ids == [1]                # 停止 token 之后的被截断
    assert core_output.new_logprobs.logprob_token_ids.shape == (1, 3)
    assert core_output.new_logprobs.logprob_token_ids[0, 0].item() == 1
    assert core_output.new_logprobs.logprobs[0, 0].item() == pytest.approx(-0.1)
