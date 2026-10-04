"""step59：GPU 批量拒绝采样（我们的 Triton 路径 vs 上游 vs Torch 参考）。

覆盖需求 059 §4 的验收点：A/B/C 逐值、全接受 / 首拒绝 / 中拒绝、B=1、D=0、全 greedy、
混合 greedy/random、零概率、one-hot q、拒绝后尾部不泄漏、p/q 分布统计、上游同层差分、
逐候选 D2H 与随机流生命周期。

**行序约定**：紧凑 logits 是 `[P+B, V]`，每请求占 `K_i` 个验证行 + **1 个 bonus 行**，
一条请求的 bonus 行紧跟在它自己的验证行后面（不是"所有 bonus 排在最后"）。下面的用例都用
"逐请求"的参数（`targets`/`p_rows`/`bonus`），由 `compact_case` 负责摆行，免得手写错位。
"""

import numpy as np
import pytest
import torch
from spec_helpers import (DRAFTS_ABC, meta_drafts, metadata_for, run_ours, run_reference,
                     run_upstream, sampling_metadata)

from minivllm.sample import expand_batch_to_tokens
from minivllm.sample import rejection_sampler as mine
from minivllm.testing.torch_rejection_sampler import torch_expand_batch_to_tokens

PLACEHOLDER = -1


def one_hot_rows(argmax_tokens, vocab_size):
    """每行一个明确的 argmax（其余位置很小），便于手算期望结果。"""
    rows = []
    for token in argmax_tokens:
        row = [0.01] * vocab_size
        row[token] = 0.9
        rows.append(row)
    return rows


def exact_one_hot(token, vocab_size):
    """严格 one-hot：Torch 参考实现"自己采 bonus"时也能确定地采到同一个 token。"""
    row = [0.0] * vocab_size
    row[token] = 1.0
    return row


def rows_for(target, num_draft):
    """`target` 给一行就重复 `num_draft` 次，给多行就逐行用（随机用例需要逐行控制）。"""
    if num_draft == 0:
        return []
    if isinstance(target[0], (list, tuple)):
        assert len(target) == num_draft, f"给了 {len(target)} 行，但草稿有 {num_draft} 枚"
        return [list(row) for row in target]
    return [list(target)] * num_draft


def compact_case(drafts, targets, bonus_tokens, *, vocab_size, probs=None, q_rows=None,
                 temperature=0.0):
    """造一个 `[P+B, V]` 的紧凑 logits 与配套张量。

    `targets[i]` 是第 i 条请求 K_i 个验证行的 argmax（greedy 用例）或概率行（`probs=True`）；
    `bonus_tokens[i]` 是它的 bonus 行（greedy 用 argmax，random 用整行概率）。
    """
    rows = []
    for draft, target, bonus in zip(drafts, targets, bonus_tokens):
        if probs:
            rows.extend(rows_for(target, len(draft)) + [list(bonus)])
        else:
            rows.extend(one_hot_rows(list(target) + [bonus], vocab_size))
    logits = torch.tensor(rows, dtype=torch.float32, device="cuda").log()
    meta = metadata_for(drafts)
    draft_probs = None
    if q_rows is not None:
        flat = [row for draft, q in zip(drafts, q_rows)
                for row in rows_for(q, len(draft))]
        draft_probs = (torch.tensor(flat, dtype=torch.float32, device="cuda")
                       if flat else None)
    if probs:
        bonus = torch.tensor([[int(np.argmax(row))] for row in bonus_tokens],
                             dtype=torch.int32, device="cuda")
    else:
        bonus = torch.tensor(bonus_tokens, dtype=torch.int32, device="cuda").unsqueeze(1)
    sm = sampling_metadata([temperature] * len(drafts), drafts)
    return meta, logits, draft_probs, bonus, sm


# ---------------------------------------------------------------------------
# A. greedy 路径
# ---------------------------------------------------------------------------


def test_greedy_mid_reject_abc(cuda_device):
    """A(3 枚，第 2 枚被拒) / B(0 枚) / C(1 枚，全接受)：逐值核对。"""
    drafts = DRAFTS_ABC
    meta, logits, _, bonus, sm = compact_case(drafts, [[1, 0, 3], [], [4]], [2, 3, 1],
                                              vocab_size=5)
    out = run_ours(meta, logits, None, bonus, sm)
    assert out.tolist() == [[1, 0, PLACEHOLDER, PLACEHOLDER],
                            [3, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER],
                            [4, 1, PLACEHOLDER, PLACEHOLDER]]
    assert out.dtype == torch.int32


def test_greedy_full_accept_appends_bonus(cuda_device):
    drafts = [[1, 2], [3]]
    meta, logits, _, bonus, sm = compact_case(drafts, [[1, 2], [3]], [0, 4], vocab_size=5)
    out = run_ours(meta, logits, None, bonus, sm)
    assert out.tolist() == [[1, 2, 0], [3, 4, PLACEHOLDER]]


def test_greedy_first_reject(cuda_device):
    drafts = [[1, 2, 3]]
    meta, logits, _, bonus, sm = compact_case(drafts, [[0, 1, 2]], [4], vocab_size=5)
    out = run_ours(meta, logits, None, bonus, sm)
    assert out.tolist() == [[0, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER]]


def test_greedy_batch_one(cuda_device):
    drafts = [[2]]
    meta, logits, _, bonus, sm = compact_case(drafts, [[2]], [1], vocab_size=5)
    out = run_ours(meta, logits, None, bonus, sm)
    assert out.tolist() == [[2, 1]]


def test_all_requests_without_drafts_is_plain_decode(cuda_device):
    """D=0：输出就是 bonus token（普通 decode 是投机的特例）。"""
    drafts = [[], []]
    meta, logits, _, bonus, sm = compact_case(drafts, [[], []], [3, 4], vocab_size=5)
    out = run_ours(meta, logits, None, bonus, sm)
    assert out.shape == (2, 1)
    assert out.tolist() == [[3], [4]]


def test_mixed_greedy_and_random(cuda_device, monkeypatch):
    """同一批里 greedy 行按 argmax 比对、random 行按 p/q 判定，互不干扰。"""
    drafts = [[1], [2]]
    # 紧凑行序：req0 验证行 + req0 bonus 行 + req1 验证行 + req1 bonus 行
    # req0（greedy）：草稿 1 == argmax 1 → 接受 + bonus 1
    # req1（random）：p=[0.1,0.1,0.8]、q=[0.05,0.05,0.9] → p[2]/q[2]=0.889 < u=0.9 → 拒绝
    rows = [[0.05, 0.9, 0.05], [0.05, 0.9, 0.05],
            [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]
    logits = torch.tensor(rows, dtype=torch.float32, device="cuda").log()
    meta = metadata_for(drafts)
    q = torch.tensor([[0.05, 0.05, 0.9]], dtype=torch.float32, device="cuda")
    bonus = torch.tensor([[1], [0]], dtype=torch.int32, device="cuda")
    sm = sampling_metadata([0.0, 1.0], drafts)
    assert (sm.all_greedy, sm.all_random) == (False, False)
    monkeypatch.setattr(mine, "generate_uniform_probs",
                        lambda *a, **k: torch.tensor([0.5, 0.9], dtype=torch.float64,
                                                     device="cuda"))
    monkeypatch.setattr(mine, "sample_recovered_tokens",
                        lambda *a, **k: torch.tensor([0, 0], dtype=torch.int32,
                                                     device="cuda"))
    out = run_ours(meta, logits, q, bonus, sm)
    assert out[0].tolist() == [1, 1]
    assert out[1].tolist() == [0, PLACEHOLDER]


# ---------------------------------------------------------------------------
# B. random 路径（注入随机数，逐值核对接受公式）
# ---------------------------------------------------------------------------


def random_case(drafts, p_rows, bonus_rows, *, q_rows=None, temperature=1.0):
    return compact_case(drafts, p_rows, bonus_rows, vocab_size=len(p_rows[0]), probs=True,
                        q_rows=q_rows, temperature=temperature)


def test_random_accept_boundary_min_1_p_over_q(cuda_device, monkeypatch):
    """`p[d]/q[d] = 0.5`：u=0.49 接受、u=0.51 拒绝（`min(1, p/q)` 的边界）。"""
    drafts = [[1]]
    meta, logits, q, bonus, sm = random_case(drafts, [[0.5, 0.5]], [[0.9, 0.1]],
                                             q_rows=[[0.5, 1.0]])
    monkeypatch.setattr(mine, "sample_recovered_tokens",
                        lambda *a, **k: torch.tensor([0], dtype=torch.int32, device="cuda"))
    monkeypatch.setattr(mine, "generate_uniform_probs",
                        lambda *a, **k: torch.tensor([0.49], dtype=torch.float64,
                                                     device="cuda"))
    assert run_ours(meta, logits, q, bonus, sm).tolist() == [[1, 0]]

    monkeypatch.setattr(mine, "generate_uniform_probs",
                        lambda *a, **k: torch.tensor([0.51], dtype=torch.float64,
                                                     device="cuda"))
    # 拒绝 → 用 recovered token（注入成 0）顶替，后面的位置不再提交
    assert run_ours(meta, logits, q, bonus, sm).tolist() == [[0, PLACEHOLDER]]


def test_random_accept_is_capped_at_one(cuda_device, monkeypatch):
    """`p/q > 1` 时接受概率是 1：u 再大也接受（`min(1, p/q)`）。"""
    drafts = [[1]]
    meta, logits, q, bonus, sm = random_case(drafts, [[0.1, 0.9]], [[0.9, 0.1]],
                                             q_rows=[[0.9, 0.1]])
    monkeypatch.setattr(mine, "generate_uniform_probs",
                        lambda *a, **k: torch.tensor([0.99999], dtype=torch.float64,
                                                     device="cuda"))
    assert run_ours(meta, logits, q, bonus, sm).tolist() == [[1, 0]]


def test_zero_draft_probability_is_rejected(cuda_device, monkeypatch):
    """q[d] == 0：防御性拒绝（`p/q` 会是 inf/NaN），走 recovered token。"""
    drafts = [[1]]
    meta, logits, q, bonus, sm = random_case(drafts, [[0.5, 0.5]], [[0.9, 0.1]],
                                             q_rows=[[1.0, 0.0]])
    monkeypatch.setattr(mine, "generate_uniform_probs",
                        lambda *a, **k: torch.tensor([0.0], dtype=torch.float64,
                                                     device="cuda"))
    monkeypatch.setattr(mine, "sample_recovered_tokens",
                        lambda *a, **k: torch.tensor([0], dtype=torch.int32, device="cuda"))
    assert run_ours(meta, logits, q, bonus, sm).tolist() == [[0, PLACEHOLDER]]


def test_point_mass_q_accepts_when_p_ge_u(cuda_device, monkeypatch):
    """`draft_probs=None`（ngram 的点质量提议）：q[d]=1，判定退化成 `p[d] >= u`。"""
    drafts = [[1]]
    meta, logits, _, bonus, sm = random_case(drafts, [[0.4, 0.6]], [[0.9, 0.1]])
    monkeypatch.setattr(mine, "sample_recovered_tokens",
                        lambda *a, **k: torch.tensor([0], dtype=torch.int32, device="cuda"))

    monkeypatch.setattr(mine, "generate_uniform_probs",
                        lambda *a, **k: torch.tensor([0.59], dtype=torch.float64,
                                                     device="cuda"))
    assert run_ours(meta, logits, None, bonus, sm).tolist() == [[1, 0]]

    monkeypatch.setattr(mine, "generate_uniform_probs",
                        lambda *a, **k: torch.tensor([0.61], dtype=torch.float64,
                                                     device="cuda"))
    assert run_ours(meta, logits, None, bonus, sm).tolist() == [[0, PLACEHOLDER]]


def test_random_full_accept_appends_bonus(cuda_device):
    """q == p 时每个位置都以概率 1 接受（u < 1）→ 追加 bonus。"""
    drafts = [[1, 2, 0]]
    p = [0.2, 0.3, 0.5]
    meta, logits, q, bonus, sm = random_case(drafts, [p], [[0.0, 0.0, 1.0]],
                                             q_rows=[p])
    out = run_ours(meta, logits, q, bonus, sm)
    assert out.tolist() == [[1, 2, 0, 2]]


def test_rejected_tail_does_not_leak(cuda_device, monkeypatch):
    """拒绝之后，后面预先算好的 recovered 值不许出现（尾部保持 -1）。"""
    drafts = [[1, 2, 3]]
    p = [0.4, 0.6]
    meta, logits, q, bonus, sm = random_case(drafts, [p], [[0.9, 0.1]],
                                             q_rows=[[0.5, 1.0]])
    # 第 1 枚被拒（u=0.9 > p[1]/q[1]=0.6），recovered = 0；后面两枚即使"能接受"也不许写进输出
    monkeypatch.setattr(mine, "generate_uniform_probs",
                        lambda *a, **k: torch.tensor([0.9, 0.0, 0.0], dtype=torch.float64,
                                                     device="cuda"))
    monkeypatch.setattr(mine, "sample_recovered_tokens",
                        lambda *a, **k: torch.tensor([0, 1, 1], dtype=torch.int32,
                                                     device="cuda"))
    out = run_ours(meta, logits, q, bonus, sm)
    assert out.tolist() == [[0, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER]]


def test_ragged_k_and_bonus_ownership(cuda_device):
    """ragged 批里每条请求的结果只落在自己那一段（K 不同也不串行）。"""
    drafts = [[1, 2], [], [3], [4, 0, 2]]
    targets = [[1, 2], [], [3], [4, 1, 2]]         # req3 第 2 枚（草稿 0）被拒 → 顶替成 1
    meta, logits, _, bonus, sm = compact_case(drafts, targets, [0, 1, 0, 3], vocab_size=5)
    out = run_ours(meta, logits, None, bonus, sm)
    assert out.tolist() == [[1, 2, 0, PLACEHOLDER],
                            [1, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER],
                            [3, 0, PLACEHOLDER, PLACEHOLDER],
                            [4, 1, PLACEHOLDER, PLACEHOLDER]]


# ---------------------------------------------------------------------------
# C. 与上游 / Torch 参考逐值差分
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drafts", [DRAFTS_ABC, [[1, 2, 3]], [[2], []], [[0, 1], [2]]])
def test_greedy_matches_upstream(cuda_device, drafts):
    """同一批张量喂给上游 `rejection_sample`，结果必须逐值相同。"""
    vocab = 5
    targets = [[(index + position) % vocab for position in range(len(draft))]
               for index, draft in enumerate(drafts)]
    bonus_tokens = [(index * 2 + 1) % vocab for index in range(len(drafts))]
    meta, logits, _, bonus, sm = compact_case(drafts, targets, bonus_tokens,
                                              vocab_size=vocab)
    ours = run_ours(meta, logits, None, bonus, sm)
    theirs = run_upstream(meta, logits, None, bonus, [0.0] * len(drafts))
    assert ours.tolist() == theirs.tolist()


@pytest.mark.parametrize("drafts", [DRAFTS_ABC, [[1, 2, 3]], [[2], []], [[0, 1], [2]]])
def test_random_matches_upstream_same_seed(cuda_device, drafts):
    """同一 seed 下（两条实现的随机数调用顺序一致）结果逐值相同。"""
    vocab = 5
    batch = len(drafts)
    rows = [[0.1 + 0.1 * ((index + token) % 3) for token in range(vocab)]
            for index in range(sum(len(d) for d in drafts) + batch)]
    meta = metadata_for(drafts)
    logits = torch.tensor(rows, dtype=torch.float32, device="cuda").log()
    probs = torch.rand(sum(len(d) for d in drafts), vocab, device="cuda")
    draft_probs = (probs / probs.sum(-1, keepdim=True)).float()
    bonus = torch.randint(0, vocab, (batch, 1), device="cuda").to(torch.int32)
    torch.manual_seed(1234)
    ours = run_ours(meta, logits, draft_probs, bonus,
                    sampling_metadata([1.0] * batch, drafts))
    torch.manual_seed(1234)
    theirs = run_upstream(meta, logits, draft_probs, bonus, [1.0] * batch)
    assert ours.tolist() == theirs.tolist()


def test_random_matches_reference_with_injected_randomness(cuda_device, monkeypatch):
    """注入同一组 u / recovered：我们的内核、上游内核、Torch 参考三者逐值相同。

    注入是 199 §8 允许的做法：把"算法错"和"随机流不同"分开——这里不一致就一定是拒绝判定
    或恢复逻辑的实现差异，而不是抽样差异。
    """
    drafts = DRAFTS_ABC
    batch = len(drafts)
    vocab = 5
    torch.manual_seed(3)
    p_rows = (torch.rand(sum(len(d) for d in drafts), vocab).softmax(-1)).tolist()
    q_rows = (torch.rand(sum(len(d) for d in drafts), vocab).softmax(-1)).tolist()
    targets, cursor = [], 0
    for draft in drafts:
        targets.append(p_rows[cursor:cursor + len(draft)])
        cursor += len(draft)
    bonus_tokens = [3, 1, 0]
    # bonus 行用严格 one-hot：参考实现是自己采 bonus 的，这样三方才会落在同一个 token 上
    bonus_rows = [exact_one_hot(token, vocab) for token in bonus_tokens]
    q_per_req, cursor = [], 0
    for draft in drafts:
        q_per_req.append(q_rows[cursor:cursor + len(draft)])
        cursor += len(draft)
    meta, logits, draft_probs, bonus, sm = compact_case(
        drafts, targets, bonus_rows, vocab_size=vocab, probs=True, q_rows=q_per_req,
        temperature=1.0)
    assert bonus.flatten().tolist() == bonus_tokens
    assert sm.all_random and not sm.all_greedy

    uniforms = torch.tensor([0.2, 0.7, 0.5, 0.9], dtype=torch.float64, device="cuda")
    recoveries = torch.tensor([3, 1, 0, 2], dtype=torch.int32, device="cuda")
    # 上游模块里的同名函数也要注入同一组随机数（否则两边抽的不是同一串）
    import vllm.v1.sample.rejection_sampler as upstream_module
    for module in (mine, upstream_module):
        monkeypatch.setattr(module, "generate_uniform_probs", lambda *a, **k: uniforms)
        monkeypatch.setattr(module, "sample_recovered_tokens", lambda *a, **k: recoveries)
    ours = run_ours(meta, logits, draft_probs, bonus, sm)
    theirs = run_upstream(meta, logits, draft_probs, bonus, [1.0] * batch)
    reference = run_reference(
        meta, logits, draft_probs, bonus,
        sampling_metadata([1.0] * batch, drafts, device="cpu"),
        uniforms=uniforms.cpu(), recoveries=recoveries.cpu())
    assert ours.tolist() == theirs.tolist() == reference.tolist()


def test_greedy_matches_reference(cuda_device):
    drafts = DRAFTS_ABC
    meta, logits, _, bonus, sm = compact_case(drafts, [[1, 0, 3], [], [4]], [2, 3, 1],
                                              vocab_size=5)
    ours = run_ours(meta, logits, None, bonus, sm)
    reference = run_reference(meta, logits, None, bonus,
                              sampling_metadata([0.0] * len(drafts), drafts, device="cpu"))
    assert ours.tolist() == reference.tolist()


def test_expand_batch_to_tokens_matches_upstream(cuda_device):
    """`expand_batch_to_tokens`（我们的 expand_kernel）与上游同名函数逐值相同。"""
    from vllm.v1.sample.rejection_sampler import expand_batch_to_tokens as upstream_expand

    values = torch.tensor([0.0, 2.0, 5.0, 1.0], device="cuda")
    cu = torch.tensor([1, 3, 7, 10], dtype=torch.int32, device="cuda")
    ours = expand_batch_to_tokens(values, cu, 10)
    theirs = upstream_expand(values, cu, 10)
    reference = torch_expand_batch_to_tokens(values, [1, 2, 4, 3])
    assert ours.tolist() == theirs.tolist() == reference.tolist()
    # replace_from/replace_to：贪心行的温度 0 → 1
    assert expand_batch_to_tokens(torch.zeros(2, device="cuda"),
                                  torch.tensor([1, 2], dtype=torch.int32, device="cuda"), 2,
                                  replace_from=0, replace_to=1).tolist() == [1.0, 1.0]


# ---------------------------------------------------------------------------
# D. 分布正确性（需求 059 §4：预先约定样本量与阈值）
# ---------------------------------------------------------------------------


def test_output_distribution_equals_target_p(cuda_device):
    """p=[0.2,0.3,0.5]、q=[0.6,0.3,0.1]：拒绝采样的**输出分布必须是 p**。

    解析（可手算，也是本用例的期望）：

        接受概率      d=0: min(1, 0.2/0.6)=1/3   d=1: min(1,1)=1   d=2: min(1,5)=1
        拒绝质量      q[0]*(1-1/3) = 0.4
        恢复分布      max(p-q,0) = [0, 0, 0.4] → 归一化后 **必然**是 token 2
        P(输出=0) = 0.6*(1/3) = 0.2     ✓
        P(输出=1) = 0.3*1     = 0.3     ✓
        P(输出=2) = 0.1*1 + 0.4 = 0.5   ✓

    样本量 N=20000、阈值 0.02：最大标准差 sqrt(0.5*0.5/20000)=0.0035，0.02 ≈ 5.7σ。
    """
    p = [0.2, 0.3, 0.5]
    q = [0.6, 0.3, 0.1]
    samples, threshold = 20000, 0.02
    drafts = [[int(token)] for token in
              np.random.default_rng(0).choice(3, size=samples, p=q)]
    meta = metadata_for(drafts)
    logits = torch.tensor([p], dtype=torch.float32, device="cuda").log().repeat(
        2 * samples, 1)
    draft_probs = torch.tensor([q], dtype=torch.float32, device="cuda").repeat(samples, 1)
    bonus = torch.tensor([[2]] * samples, dtype=torch.int32, device="cuda")
    sm = sampling_metadata([1.0] * samples, drafts)
    out = run_ours(meta, logits, draft_probs, bonus, sm)[:, 0].cpu().numpy()
    frequencies = np.bincount(out, minlength=3) / samples
    assert np.all(np.abs(frequencies - np.array(p)) < threshold), frequencies.tolist()


def test_recovered_distribution_is_max_p_minus_q(cuda_device):
    """恢复分布 ∝ `max(p-q, 0)`：p=[0.3,0.3,0.4]、q=[0.6,0.1,0.3] → token1:token2 = 2:1。

    样本量 N=30000、阈值 0.015：2/3 的标准差 sqrt(2/9/30000)=0.0027，0.015 ≈ 5.5σ。
    """
    p = [0.3, 0.3, 0.4]
    q = [0.6, 0.1, 0.3]
    samples, threshold = 30000, 0.015
    meta = metadata_for([[0]] * samples)
    target_probs = torch.tensor([p], dtype=torch.float32, device="cuda").repeat(samples, 1)
    draft_probs = torch.tensor([q], dtype=torch.float32, device="cuda").repeat(samples, 1)
    sm = sampling_metadata([1.0] * samples, [[0]] * samples)
    recovered = mine.sample_recovered_tokens(
        1, meta.num_draft_tokens, meta.cu_num_draft_tokens, meta.draft_token_ids,
        draft_probs, target_probs, sm, "cuda").cpu().numpy()
    frequencies = np.bincount(recovered, minlength=3) / samples
    assert frequencies[0] == 0.0
    assert abs(frequencies[1] - 2 / 3) < threshold, frequencies.tolist()
    assert abs(frequencies[2] - 1 / 3) < threshold, frequencies.tolist()


# ---------------------------------------------------------------------------
# E. parse_output / 随机流生命周期 / 逐候选 D2H
# ---------------------------------------------------------------------------


def test_parse_output_filters_placeholder_and_discard(cuda_device):
    from minivllm.sample import RejectionSampler

    padded = torch.tensor([[1, 0, PLACEHOLDER, PLACEHOLDER],
                           [3, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER],
                           [4, 5, PLACEHOLDER, PLACEHOLDER]], dtype=torch.int32,
                          device="cuda")
    outputs, logprobs = RejectionSampler.parse_output(padded, vocab_size=10)
    assert outputs == [[1, 0], [3], [4, 5]]
    assert logprobs is None
    outputs, _ = RejectionSampler.parse_output(padded, vocab_size=10, discard_req_indices=[0])
    assert outputs == [[], [3], [4, 5]]


def test_parse_output_drops_out_of_vocab_ids(cuda_device):
    """`id >= vocab_size` 是脏数据（上游同一个条件），必须被过滤掉。"""
    from minivllm.sample import RejectionSampler

    padded = torch.tensor([[1, 99, 2, PLACEHOLDER]], dtype=torch.int32, device="cuda")
    outputs, _ = RejectionSampler.parse_output(padded, vocab_size=10)
    assert outputs == [[1, 2]]


def test_zero_draft_requests_do_not_consume_randomness(cuda_device):
    """K=0 的请求不推进随机流（上游同款；否则同 seed 下结果会随批组成变化）。"""
    generator = torch.Generator(device="cuda").manual_seed(7)
    p = [0.5, 0.5]
    # 批 A：[K=1, K=0]
    meta = metadata_for([[1], []])
    logits = torch.tensor([p], dtype=torch.float32, device="cuda").log().repeat(3, 1)
    q = torch.tensor([p], dtype=torch.float32, device="cuda")
    bonus = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    sm = sampling_metadata([1.0, 1.0], [[1], []], generators={0: generator})
    torch.manual_seed(11)
    out_with_empty = run_ours(meta, logits, q, bonus, sm).cpu()
    # 批 B：只有那条 K=1 的请求
    meta2 = metadata_for([[1]])
    logits2 = torch.tensor([p], dtype=torch.float32, device="cuda").log().repeat(2, 1)
    bonus2 = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    sm2 = sampling_metadata([1.0], [[1]], generators={0: generator})
    torch.manual_seed(11)
    out_alone = run_ours(meta2, logits2, q, bonus2, sm2).cpu()
    assert out_with_empty[0].tolist() == out_alone[0].tolist()


def test_no_per_candidate_device_syncs(cuda_device):
    """验证路径不能有"逐候选 D2H"：两种批大小的 D2H 次数都为 0（不随 B×K 增长）。"""
    from torch.profiler import ProfilerActivity, profile

    def d2h_per_step(drafts):
        batch = len(drafts)
        meta = metadata_for(drafts)
        rows = [[0.2, 0.3, 0.5]] * (sum(len(d) for d in drafts) + batch)
        logits = torch.tensor(rows, dtype=torch.float32, device="cuda").log()
        q = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32,
                         device="cuda").repeat(sum(len(d) for d in drafts), 1)
        bonus = torch.tensor([[2]] * batch, dtype=torch.int32, device="cuda")
        sm = sampling_metadata([1.0] * batch, drafts)
        for _ in range(3):
            run_ours(meta, logits, q, bonus, sm)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                run_ours(meta, logits, q, bonus, sm)
            torch.cuda.synchronize()
        return sum(1 for event in prof.events() if "DtoH" in event.name) / 5

    assert d2h_per_step([[1]]) == 0
    assert d2h_per_step([[1, 2, 3, 4, 5]] * 8) == 0


def test_draft_slice_helper(cuda_device):
    assert meta_drafts(metadata_for([[1, 2, 3], [], [4]])) == [[1, 2, 3], [], [4]]
