"""step59：接受率统计、采样器侧的投机语义、以及端到端（tiny 模型）行为。

- `SpecDecodingStats`：只统计**已经验证过**的候选（提议数不能当接受数）。
- `Sampler`：`predict_bonus_token` 的历史口径、投机版 min_tokens 屏蔽。
- 端到端：真 tiny 模型上 greedy 投机 == 非投机（draft_model 与 ngram 两条提议路径）。
"""

import torch
from helpers import make_engine, run_prompts, sampling_metadata

from minivllm import SpeculativeConfig
from minivllm.sample import Sampler
from minivllm.sample import rejection_sampler as mine
from minivllm.spec_decode import SpecDecodingStats

PROMPTS = (("A", [1, 2, 3, 4, 5, 6]), ("B", [2, 3, 4, 5, 6, 7, 8]))
PATTERN = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# SpecDecodingStats
# ---------------------------------------------------------------------------


def test_stats_new_and_observe():
    stats = SpecDecodingStats.new(3)
    assert stats.num_spec_tokens == 3
    assert stats.num_accepted_tokens_per_pos == [0, 0, 0]
    assert stats.num_draft_tokens_per_pos == [0, 0, 0]
    # 一条请求：验证 3 枚、接受 2 枚
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    # 另一条：验证 1 枚、接受 1 枚（只采用/验证了 1 枚，不能拿它"提议了 3 枚"来记）
    stats.observe_draft(num_draft_tokens=1, num_accepted_tokens=1)
    assert (stats.num_drafts, stats.num_draft_tokens, stats.num_accepted_tokens) == (2, 4, 3)
    assert stats.num_accepted_tokens_per_pos == [2, 1, 0]
    assert stats.num_draft_tokens_per_pos == [2, 1, 1]


def test_stats_rejects_more_accepted_than_allowed():
    """接受的枚数不可能超过配置的 K（超了说明把未验证的也算进来了）。"""
    stats = SpecDecodingStats.new(2)
    try:
        stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=3)
    except AssertionError:
        return
    raise AssertionError("接受数 > num_spec_tokens 时必须断言失败")


def test_rejection_sample_method_only_standard():
    """本关只接 standard：synthetic / block 在 75 关，遇到就明确报错。"""
    for method in ("synthetic", "block"):
        try:
            SpeculativeConfig(method="ngram", num_speculative_tokens=1,
                              rejection_sample_method=method)
        except ValueError as error:
            assert "75" in str(error)
            continue
        raise AssertionError(f"rejection_sample_method={method!r} 应该被拒绝")


def test_rejection_sampler_rejects_non_standard_config():
    sampler = Sampler()
    for method in ("synthetic", "block"):
        spec = SpeculativeConfig(method="ngram", num_speculative_tokens=1)
        object.__setattr__(spec, "rejection_sample_method", method)
        try:
            mine.RejectionSampler(sampler, spec)
        except ValueError as error:
            assert "standard" in str(error)
            continue
        raise AssertionError(f"{method!r} 不该被 RejectionSampler 接受")
    # standard（以及没有 spec_config）必须能建出来
    mine.RejectionSampler(sampler, SpeculativeConfig(method="ngram", num_speculative_tokens=1))
    mine.RejectionSampler(sampler)


# ---------------------------------------------------------------------------
# Sampler 侧的投机语义
# ---------------------------------------------------------------------------


def test_predict_bonus_token_uses_all_drafts_as_history():
    """bonus 行的惩罚历史 = 已提交 + **全部**草稿（能走到 bonus 就说明草稿都被接受了）。"""
    sampler = Sampler()
    logits = torch.zeros(1, 4)
    sm = sampling_metadata([1.0], [[1, 2]], device="cpu", output_token_ids=[[3]],
                           no_penalties=False, frequency_penalties=[1.0],
                           presence_penalties=[0.0], repetition_penalties=[1.0])

    # 历史 = [3]（不带草稿）→ 只有 token 3 被罚（frequency penalty 各减 1）
    without = sampler.apply_logits_processors(logits.clone(), sm)
    assert without.tolist() == [[0.0, 0.0, 0.0, -1.0]]

    # 历史 = [3, 1, 2] → 草稿也被算进去：token 1/2/3 各减 1
    with_drafts = sampler.apply_logits_processors(logits.clone(), sm,
                                                  predict_bonus_token=True)
    assert with_drafts.tolist() == [[0.0, -1.0, -1.0, -1.0]]


def test_min_tokens_spec_decode_masks_leading_rows():
    """投机版 min_tokens：每条请求只屏蔽**前 n_mask 行**（上游 `apply_with_spec_decode`）。

    算例（上游注释）：`num_draft_tokens=[2,3,1]` → `cumsum=[0,2,5,6]`。
    """
    sampler = Sampler()
    num_draft_tokens = [2, 3, 1]
    logits = torch.zeros(6, 10)
    # 第 0 条：min_tokens=3、已提交 1 个 → remaining=2 → 前 2 行（0,1）屏蔽 token 7
    # 第 1 条：min_tokens=0 → 不屏蔽
    # 第 2 条：min_tokens=2、已提交 1 个 → remaining=1 → 第 5 行屏蔽 token 9
    sm = sampling_metadata([1.0, 1.0, 1.0], [[1, 2], [3, 4, 5], [6]], device="cpu",
                           output_token_ids=[[9], [9, 9, 9, 9, 9], [9]],
                           min_tokens=[3, 0, 2], stop_token_ids=[[7], [7], [9]])
    out = sampler.apply_min_tokens_for_spec_decode(logits.clone(), sm, num_draft_tokens)
    masked = {(row, token) for row in range(6) for token in range(10)
              if out[row, token] == float("-inf")}
    assert masked == {(0, 7), (1, 7), (5, 9)}


def test_min_tokens_plain_path_masks_whole_row():
    """非投机路径：min_tokens 只管"这一行"，一次 index_put_ 处理整批。"""
    sampler = Sampler()
    logits = torch.zeros(2, 10)
    sm = sampling_metadata([1.0, 1.0], [[], []], device="cpu",
                           output_token_ids=[[9], [9, 9, 9]],
                           min_tokens=[3, 0], stop_token_ids=[[7, 8], [7, 8]])
    out = sampler.apply_min_tokens(logits.clone(), sm)
    masked = {(row, token) for row in range(2) for token in range(10)
              if out[row, token] == float("-inf")}
    assert masked == {(0, 7), (0, 8)}


# ---------------------------------------------------------------------------
# 端到端（真 tiny 模型、GPU）
# ---------------------------------------------------------------------------


def test_greedy_speculative_matches_non_speculative_draft_model(cuda_device, tiny_dir,
                                                                hf_config):
    plain, _ = run_prompts(tiny_dir=tiny_dir, hf_config=hf_config, prompts=PROMPTS,
                           spec_k=None, max_tokens=8)
    spec, stats = run_prompts(tiny_dir=tiny_dir, hf_config=hf_config, prompts=PROMPTS,
                              spec_k=3, max_tokens=8, collect_stats=True)
    assert spec == plain
    observed = [entry for entry in stats if entry is not None]
    assert observed, "开投机之后必须至少有一轮统计"
    assert sum(entry.num_draft_tokens for entry in observed) > 0
    # 统计的是"验证过的候选"：绝不超过配置的 K × 轮数
    assert all(entry.num_draft_tokens <= 3 * entry.num_drafts for entry in observed)


def test_greedy_speculative_matches_non_speculative_ngram(cuda_device, tiny_dir, hf_config):
    """ngram 提议（点质量 q、draft_probs=None）也要与非投机一致。"""
    prompts = (("A", PATTERN),)
    plain, _ = run_prompts(tiny_dir=tiny_dir, hf_config=hf_config, prompts=prompts,
                           spec_k=None, method="ngram", max_tokens=8)
    spec, stats = run_prompts(tiny_dir=tiny_dir, hf_config=hf_config, prompts=prompts,
                              spec_k=3, method="ngram", max_tokens=8, collect_stats=True)
    assert spec == plain
    assert sum(entry.num_draft_tokens for entry in stats if entry is not None) > 0


def test_random_speculative_runs_and_reports_stats(cuda_device, tiny_dir, hf_config):
    """随机采样（q == p，因为 draft 与 target 是同一个 tiny 模型）也要跑通并统计接受率。"""
    outputs, stats = run_prompts(tiny_dir=tiny_dir, hf_config=hf_config, prompts=PROMPTS,
                                 spec_k=3, max_tokens=6, temperature=1.0, seed=7,
                                 collect_stats=True)
    assert all(len(tokens) == 6 for tokens in outputs.values())
    observed = [entry for entry in stats if entry is not None]
    assert observed and all(entry.num_accepted_tokens <= entry.num_draft_tokens
                            for entry in observed)


def test_scheduler_stats_are_fed_by_verified_outputs(cuda_device, tiny_dir, hf_config):
    """`make_spec_decoding_stats` 的入参必须来自"本轮真取到结果"的那些请求。"""
    engine, core, _runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3)
    from minivllm import SamplingParams

    calls = []
    original = core.scheduler.make_spec_decoding_stats

    def spy(stats, num_draft_tokens, num_accepted_tokens):
        calls.append((num_draft_tokens, num_accepted_tokens))
        return original(stats, num_draft_tokens, num_accepted_tokens)

    core.scheduler.make_spec_decoding_stats = spy
    for req_id, prompt in PROMPTS:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
    for _ in range(40):
        if not engine.has_unfinished_requests():
            break
        engine.step()
    engine.shutdown()

    assert calls, "开着投机跑完必须有过验证"
    assert all(0 <= accepted <= drafted <= 3 for drafted, accepted in calls)
    assert any(accepted > 0 for _drafted, accepted in calls), "这里必然有被接受的草稿"
