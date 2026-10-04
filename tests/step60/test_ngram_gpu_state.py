"""step60：GPU ngram 提议者的状态维护（需求 060 §2/§3.3-§3.5/§4）。

三块：
  A. 匹配/提取与上游冻结实现一致 + "宽度 vs 有效个数"的语义；
  B. GPU 历史的增量维护（scatter 新 token、**不写回**权威长度、行重排、只拷新增）；
  C. 调度侧对齐（`update_scheduler_for_invalid_drafts`：占位裁到有效、`-1` 永远出不去）。
"""

import dataclasses

import numpy as np
import pytest
import torch
from ngram_helpers import FakeInputBatch, make_engine, make_rows, run_prompts
from vllm.v1.spec_decode.ngram_proposer import (
    _find_longest_matched_ngram_and_propose_tokens as upstream_find)

from minivllm import SamplingParams, SpeculativeConfig
from minivllm.spec_decode.ngram_proposer_gpu import (NgramGPUKernel, NgramProposerGPU,
                                                     update_ngram_gpu_tensors_incremental,
                                                     update_scheduler_for_invalid_drafts)

MAX_MODEL_LEN = 64


class _Config:
    def __init__(self, k=3, min_n=2, max_n=4, max_model_len=MAX_MODEL_LEN, max_num_seqs=4):
        self.speculative_config = SpeculativeConfig(
            method="ngram_gpu", num_speculative_tokens=k,
            prompt_lookup_min=min_n, prompt_lookup_max=max_n)
        self.model_config = type("M", (), {"max_model_len": max_model_len})()
        self.scheduler_config = type("S", (), {"max_num_seqs": max_num_seqs})()


def make_gpu_proposer(device="cpu", **kwargs) -> NgramProposerGPU:
    """CPU 上也能跑：`NgramGPUKernel` 全程是 torch 张量运算（上游那版要 CUDA + torch.compile）。"""
    return NgramProposerGPU(_Config(**kwargs), torch.device(device))


def make_state(proposer, histories: list[list[int]], device="cpu"):
    """造一份"显存历史 + 长度表"（形状与 Runner 里那两个缓冲一致）。"""
    batch = FakeInputBatch(histories, proposer.max_model_len, dtype=torch.int32)
    token_ids = torch.zeros(batch.max_num_reqs, proposer.max_model_len, dtype=torch.int32,
                            device=device)
    lengths = torch.zeros(batch.max_num_reqs, dtype=torch.int32, device=device)
    for index, tokens in enumerate(histories):
        token_ids[index, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32,
                                                      device=device)
        lengths[index] = len(tokens)
    return batch, token_ids, lengths


# ---------------------------------------------------------------------------
# A. 匹配/提取 + 有效个数
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [
    [1, 2, 3, 1, 2, 9, 1, 2],                  # 多处等长匹配
    [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
    [5, 5, 5, 5, 5, 5],                        # 重复全同
    [7, 8, 9, 1, 2, 3, 7, 8, 9],
    [1, 2],                                    # 太短
    [9, 9, 9, 1, 9, 9, 9],
])
def test_gpu_matcher_matches_upstream(device, tokens):
    """同一段历史：GPU kernel 与上游冻结的 CPU 参考实现逐值一致（长度小于 max_model_len）。"""
    proposer = make_gpu_proposer(device)
    batch, token_ids, lengths = make_state(proposer, [tokens], device)
    combined = torch.ones(1, dtype=torch.bool, device=device)
    drafts, num_valid = proposer.kernel(lengths, token_ids, combined)
    expected = upstream_find(np.array(tokens, dtype=np.int32), proposer.min_n,
                             proposer.max_n, 10 ** 9, proposer.k).tolist()
    assert drafts[0].tolist()[:len(expected)] == expected
    assert int(num_valid[0]) == len(expected)
    # 有效个数之后的格子必须是 -1（占位）
    assert set(drafts[0].tolist()[len(expected):]) <= {-1}


def test_width_versus_valid_count(device):
    """宽度是 `k`，有效数才是语义：`[1,2,9,1,2]` 的 n=2 匹配点后面只剩 3 个 token。"""
    proposer = make_gpu_proposer(device, k=4, min_n=2, max_n=2)
    _batch, token_ids, lengths = make_state(proposer, [[1, 2, 9, 1, 2]], device)
    drafts, num_valid = proposer.kernel(lengths, token_ids,
                                        torch.ones(1, dtype=torch.bool, device=device))
    assert drafts.shape == (1, 4)                     # 固定宽度 4
    assert drafts[0].tolist() == [9, 1, 2, -1]        # 只抄到 3 枚，第 4 格是占位
    assert int(num_valid[0]) == 3                     # "有效个数"才是语义
    # 与 CPU 参考实现比：它返回的就是那 3 枚
    assert upstream_find(np.array([1, 2, 9, 1, 2], dtype=np.int32), 2, 2, 10 ** 9,
                         4).tolist() == [9, 1, 2]


def test_gpu_does_not_cap_by_remaining_model_length(device):
    """与上游 CPU 版的**已知差异**：GPU 不从历史里"抄"超越上限的草稿，所以不做
    `k = min(k, max_model_len - total)` 那个上限（上游 GPU 版同样没有）。"""
    tokens = [1, 2, 3, 1, 2, 3]
    proposer = make_gpu_proposer(device, k=2, min_n=3, max_n=3, max_model_len=len(tokens))
    _batch, token_ids, lengths = make_state(proposer, [tokens], device)
    drafts, num_valid = proposer.kernel(lengths, token_ids,
                                        torch.ones(1, dtype=torch.bool, device=device))
    assert int(num_valid[0]) > 0
    # CPU 版在同样的长度下会被 max_model_len 上限压成 0 个
    assert upstream_find(np.array(tokens, dtype=np.int32), 3, 3, len(tokens), 2).tolist() == []


# ---------------------------------------------------------------------------
# B. GPU 历史：scatter / 只读长度 / 行重排
# ---------------------------------------------------------------------------


def test_propose_scatters_new_tokens_and_keeps_lengths_read_only(device):
    """`propose()` 把新采样的 token 写进历史，但**不改** `num_tokens_no_spec`（只读输入）。"""
    proposer = make_gpu_proposer(device, k=2, min_n=2, max_n=2)
    _batch, token_ids, lengths = make_state(proposer, [[1, 2, 1, 2, 1, 2]], device)
    before_lengths = lengths.clone()
    before_row = token_ids[0].clone()

    sampled = torch.tensor([[9, 9]], dtype=torch.int32, device=device)   # 本轮采样 2 个
    counts = torch.tensor([2], dtype=torch.int32, device=device)
    drafts, num_valid = proposer.propose(2, lengths, token_ids, sampled, counts)

    # 新 token 落在 num_tokens_no_spec 之后；历史其余部分没动
    assert token_ids[0, 6:8].tolist() == [9, 9]
    assert token_ids[0, :6].tolist() == before_row[:6].tolist()
    assert token_ids[0, 8:].tolist() == before_row[8:].tolist()
    # 长度表是只读的：写完历史也不会被改（否则同一个输出会被累计两次）
    assert lengths.tolist() == before_lengths.tolist()
    # 后缀变成 [9,9]，更早出现过吗？没有 → 不提
    assert int(num_valid[0]) == 0
    assert drafts[0].tolist() == [-1, -1]


def test_propose_uses_temp_length_for_matching(device):
    """匹配用的是 `num_tokens_tmp = 长度 + 本轮有效采样数`，不是历史长度。"""
    proposer = make_gpu_proposer(device, k=2, min_n=2, max_n=2)
    _batch, token_ids, lengths = make_state(proposer, [[3, 4, 3, 4, 3, 4]], device)
    sampled = torch.tensor([[3, 4]], dtype=torch.int32, device=device)
    counts = torch.tensor([2], dtype=torch.int32, device=device)
    drafts, num_valid = proposer.propose(2, lengths, token_ids, sampled, counts)
    # 历史变成 [3,4,3,4,3,4,3,4]：后缀 "3 4" 最早出现在第 0 位，后面是 3,4 → 草稿 [3,4]
    assert drafts[0].tolist() == [3, 4] and int(num_valid[0]) == 2


def test_propose_asserts_fixed_k(device):
    """上游对 K 有固定值断言（动态 K 属 71 关，不能在这里偷偷放开）。"""
    proposer = make_gpu_proposer(device, k=3)
    _batch, token_ids, lengths = make_state(proposer, [[1, 2, 1, 2]], device)
    with pytest.raises(AssertionError):
        proposer.propose(2, lengths[:1], token_ids[:1],
                         torch.tensor([[1]], dtype=torch.int32, device=device),
                         torch.tensor([1], dtype=torch.int32, device=device))


def test_update_token_ids_ngram_masks_discarded_and_invalid(device):
    """丢弃的请求整行 -1；越界 id 不计入有效数；没有有效采样时回退历史末尾那个 token。

    有效采样按**前导前缀**解释（采样器给的 id 必然 `< vocab_size`，哨兵只出现在尾部）：
    计数按 `< vocab_size` 过滤，scatter 只写前 `count` 个。
    """
    proposer = make_gpu_proposer(device, k=2)
    batch = FakeInputBatch([[1, 2, 3], [4, 5, 6]], proposer.max_model_len, vocab_size=10)
    _b, token_ids, lengths = make_state(proposer, [[1, 2, 3], [4, 5, 6]], device)

    sampled = torch.tensor([[7, 8], [5, 99]], dtype=torch.int32, device=device)  # 第二行尾巴越界
    discard = torch.tensor([True, False], dtype=torch.bool, device=device)
    next_token_ids, counts, valid_ids = proposer.update_token_ids_ngram(
        sampled, batch, token_ids, lengths, discard)

    assert counts.tolist() == [0, 1]                      # 第一行丢弃 → 0；第二行只有 5 有效
    assert valid_ids[0].tolist() == [-1, -1]
    assert valid_ids[1].tolist() == [5, 99]               # 越界只影响计数，不清零（上游同款）
    # 没有有效采样 → 回退历史末尾那个 token（3 / 5）
    assert next_token_ids.tolist() == [3, 5]
    # 之后 `propose` 只会把**前 count 个**写进历史 → 尾巴上的 99 不会污染历史
    proposer.propose(2, lengths[:2], token_ids[:2], valid_ids, counts)
    assert token_ids[1, :7].tolist() == [4, 5, 6, 5, 0, 0, 0]


def test_history_follows_request_ids_across_reorder(device):
    """批序列 `[A,B,C] → [C,A] → [D,C,A]`：历史/长度必须跟着**请求**走，不跟着行号。"""
    proposer = make_gpu_proposer(device, k=2, min_n=1, max_n=1, max_num_seqs=4)
    batch = FakeInputBatch([], proposer.max_model_len, dtype=torch.int32, max_num_reqs=4)
    token_ids = torch.zeros(batch.max_num_reqs, proposer.max_model_len, dtype=torch.int32,
                            device=device)
    lengths = torch.zeros(batch.max_num_reqs, dtype=torch.int32, device=device)

    def rebuild(rows: list[tuple[str, list[int]]], prev: dict[str, int] | None,
                new_ids: set[str]):
        batch.req_ids = [req_id for req_id, _ in rows]
        batch.req_id_to_index = {req_id: index for index, (req_id, _) in enumerate(rows)}
        for index, (_req_id, tokens) in enumerate(rows):
            batch.set_row(index, tokens, req_id=_req_id)
        update_ngram_gpu_tensors_incremental(batch, token_ids, lengths, new_ids, prev, device)
        return {req_id: index for index, (req_id, _) in enumerate(rows)}

    # 第一轮：A/B/C 各自的历史
    state_a = rebuild([("A", [1, 1, 1]), ("B", [2, 2, 2]), ("C", [3, 3, 3])], None,
                      {"A", "B", "C"})
    assert token_ids[0, :3].tolist() == [1, 1, 1]
    assert token_ids[1, :3].tolist() == [2, 2, 2]

    # 第二轮：B 结束 → C 压到行 0，A 到行 1（历史整行搬家）
    rebuild([("C", [3, 3, 3]), ("A", [1, 1, 1])], state_a, set())
    assert token_ids[0, :3].tolist() == [3, 3, 3]        # 行 0 现在是 C
    assert token_ids[1, :3].tolist() == [1, 1, 1]        # 行 1 现在是 A
    assert lengths[:2].tolist() == [3, 3]

    # 第三轮：D 进来（整段拷一次）→ [D, C, A]
    rebuild([("D", [4, 4, 4, 4]), ("C", [3, 3, 3]), ("A", [1, 1, 1])],
            {"C": 0, "A": 1}, {"D"})
    assert token_ids[0, :4].tolist() == [4, 4, 4, 4]
    assert token_ids[1, :3].tolist() == [3, 3, 3]
    assert token_ids[2, :3].tolist() == [1, 1, 1]
    assert lengths[:3].tolist() == [4, 3, 3]

    # 每行请求自己的候选：D 的历史 [4,4,4,4] + 采样 4 → 后缀 "4" 匹配到最早那处 → [4]
    sampled = torch.tensor([[4, -1], [-1, -1], [-1, -1]], dtype=torch.int32, device=device)
    counts = torch.tensor([1, 0, 0], dtype=torch.int32, device=device)
    drafts, num_valid = proposer.propose(2, lengths[:3], token_ids[:3], sampled, counts)
    # D 的历史 [4,4,4,4] + 采样 4 → n=1 时后缀 "4" 最早在第 0 位，后面还有 4 个 → 抄满 2 枚
    assert drafts[0].tolist() == [4, 4] and int(num_valid[0]) == 2
    assert int(num_valid[1]) == 0 and int(num_valid[2]) == 0


def test_incremental_update_does_not_recopy_whole_history(device):
    """第二步之后只搬"新增 token"：用 profiler 量 H2D 字节数，不随 max_model_len 增长。"""
    from torch.profiler import ProfilerActivity, profile

    def upload_bytes(max_model_len):
        proposer = make_gpu_proposer(device, max_model_len=max_model_len)
        batch = FakeInputBatch([[1, 2, 3, 4] * 4], max_model_len, dtype=torch.int32)
        token_ids = torch.zeros(batch.max_num_reqs, max_model_len, dtype=torch.int32,
                                device=device)
        lengths = torch.zeros(batch.max_num_reqs, dtype=torch.int32, device=device)
        prev = {"r0": 0}
        update_ngram_gpu_tensors_incremental(batch, token_ids, lengths, {"r0"}, None, device)
        # 更新历史末尾一个 token（模拟本轮采样），再增量同步
        batch.set_row(0, [1, 2, 3, 4] * 4 + [7], req_id="r0")
        torch.cuda.synchronize() if device == "cuda" else None
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            update_ngram_gpu_tensors_incremental(batch, token_ids, lengths, set(), prev,
                                                 device)
            torch.cuda.synchronize() if device == "cuda" else None
        total = 0
        for event in prof.events():
            if "Memcpy" in event.name and ("HtoD" in event.name):
                total += event.cpu_memory_usage or 0
        return total, lengths[0].item()

    small, length_small = upload_bytes(64)
    big, length_big = upload_bytes(512)
    assert length_small == length_big == 17
    # 第二步只同步长度 + 新增 token：字节数与 max_model_len 无关（不是"整段历史"）
    assert big <= max(small, 4096), (small, big)


# ---------------------------------------------------------------------------
# C. 调度侧对齐：占位裁到有效
# ---------------------------------------------------------------------------


def test_update_scheduler_for_invalid_drafts_trims_and_filters():
    assert update_scheduler_for_invalid_drafts([1, 2, 3, -1], 2) == [1, 2]
    assert update_scheduler_for_invalid_drafts([1, 2, 3, -1], 4) == [1, 2, 3]   # -1 被滤掉
    assert update_scheduler_for_invalid_drafts([1, 2, 3], 99) == [1, 2, 3]      # 上限夹住
    assert update_scheduler_for_invalid_drafts([1, 2, 3], 0) == []
    assert update_scheduler_for_invalid_drafts([1, 2, 3], -5) == []
    assert update_scheduler_for_invalid_drafts([1, 2, 3], None) == [1, 2, 3]    # CPU 提议者


def test_scheduler_never_stores_sentinel(cuda_device, tiny_dir, hf_config):
    """端到端：Scheduler 收下草稿之后，`request.spec_token_ids` 里不许出现 -1。"""
    engine, core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                       method="ngram_gpu", device=cuda_device,
                                       max_num_seqs=2)
    seen = []
    scheduler = core.scheduler
    original = scheduler.update_draft_token_ids

    def spy(drafts):
        original(drafts)
        for request in list(scheduler.running) + list(scheduler.waiting):
            seen.append(list(request.spec_token_ids))
        return None

    scheduler.update_draft_token_ids = spy
    engine.add_request("A", [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                       SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
    for _ in range(30):
        if not engine.has_unfinished_requests():
            break
        engine.step()
    engine.shutdown()

    assert seen, "至少要有一轮草稿交接"
    assert all(token >= 0 for row in seen for token in row), seen
    assert any(row for row in seen), "这条 prompt 有重复片段，应该真的提出过草稿"


def test_d2h_is_independent_of_history_length(cuda_device):
    """需求 §4："GPU 分支不能每轮整段 D2H"——`propose()` 的 D2H 字节数只跟 B×K 有关。"""
    from torch.profiler import ProfilerActivity, profile

    def d2h_bytes(max_model_len):
        proposer = make_gpu_proposer(cuda_device, k=3, min_n=1, max_n=3,
                                     max_model_len=max_model_len)
        history = ([1, 2, 3, 4] * (max_model_len // 4))[:max_model_len // 2]
        _batch, token_ids, lengths = make_state(proposer, [history], cuda_device)
        sampled = torch.tensor([[7]], dtype=torch.int32, device=cuda_device)
        counts = torch.tensor([1], dtype=torch.int32, device=cuda_device)
        proposer.propose(3, lengths, token_ids, sampled, counts)     # 预热
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            proposer.propose(3, lengths, token_ids, sampled, counts)
            torch.cuda.synchronize()
        return sum(1 for event in prof.events() if "DtoH" in event.name)

    assert d2h_bytes(64) == d2h_bytes(1024) == 0, "提议本身不该有 D2H（草稿由调用方一次性取回）"
