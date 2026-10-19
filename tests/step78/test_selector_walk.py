"""step78：DFlash2 的候选关联与条件 walk。

五条线（需求 078 §5）：

    A. **公式层差分**：分组卷积 / 边分 / 候选 top-k 与显式循环参考实现逐值相等；并与
       **上游真实的 `DFlashGroupedConv` / `CandidateSelector` / `_score_edges` 模块**对照
       （只允许测试 import 上游；生产路径不 import）。
    B. **第一步选择影响第二步**：手写边分让"第一步选 A 还是 B"决定第二步的最优候选，
       逐步比较生产内核、参考 walk、上游内核。
    C. **与上游内核逐值差分**：`_selector_walk_kernel` 在 greedy / 概率、多种 K/top-k 下
       的 token 与 realized scores 与上游**逐位相等**。
    D. **槽位/补齐/重排**：B 重排、slot 复用、`sample_idx_mapping=-1` 的补齐行不写任何东西。
    E. **噪声按 token id 取**：同一组 token 换列排布后，采样结果跟着 token 走（不跟列号走）。
"""

import pytest
import torch

from dflash2_helpers import (DEVICE, make_speculator, make_upstream_conv,
                             make_upstream_selector, set_walk_inputs, walk_case)
from minivllm.models.qwen3_dflash2 import _grouped_conv, _score_edges
from minivllm.testing.torch_dflash2 import (reference_greedy_walk,
                                            reference_grouped_conv,
                                            reference_score_edges)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="候选 walk 是 Triton 内核（上游同款边界），只在 CUDA 上可执行")

K = 3
TOP_K = 3
VOCAB = 11


# ---------------- A. 公式层差分 ----------------

@pytest.mark.parametrize("block_size", [4, 5])
def test_grouped_conv_matches_the_explicit_reference(block_size):
    """分组卷积与逐 token 定义的参考实现逐值相等（含非 2 的幂的 block_size）。"""
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(12, 32, generator=generator)
    delta = torch.randn(12, 3, 4, generator=generator)
    base = torch.randn(3, 32, generator=generator)
    ours = _grouped_conv(hidden, delta, base, block_size, 4, 8, 3)
    reference = reference_grouped_conv(hidden, delta, base, block_size, 4, 8, 3)
    assert torch.equal(ours, reference), "向量化写法与逐 token 公式不一致"


def test_grouped_conv_does_not_look_across_query_blocks():
    """`block_size` 的边界：块首两行不能回看到上一块的尾巴。

    位置取模之后 `position >= tap` 才累加。取 `block_size=4, taps=3`：
    第 4 行（块首）只能看自己；第 5 行能看 4/5；第 6 行能看 4/5/6。
    """
    hidden = torch.zeros(8, 32)
    for token in range(8):
        hidden[token] = float(token + 1)
    delta = torch.zeros(8, 3, 4)
    base = torch.ones(3, 32)
    out = _grouped_conv(hidden, delta, base, 4, 4, 8, 3)
    # 块 0：行 0 = 1、行 1 = 1+2、行 2 = 1+2+3、行 3 = 2+3+4（只累加 taps 项）
    assert out[0, 0].item() == 1.0
    assert out[1, 0].item() == 3.0
    assert out[2, 0].item() == 6.0
    assert out[3, 0].item() == 9.0
    # 块 1：行 4 只能看自己（=5），绝不能是 2+3+4+5=14 那种跨块结果
    assert out[4, 0].item() == 5.0
    assert out[5, 0].item() == 11.0
    assert out[6, 0].item() == 18.0   # 5+6+7（只看块内 4/5/6）


def test_score_edges_matches_the_explicit_reference():
    """`[B,K,k,k]` 三元边分与逐元素循环参考实现一致。"""
    generator = torch.Generator().manual_seed(2)
    batch, steps, top_k, rank, hidden = 3, 4, 3, 3, 5
    candidate_ids = torch.stack([
        torch.randperm(VOCAB, generator=generator)[:top_k] for _ in range(batch * steps)
    ]).view(batch, steps, top_k)
    unary = torch.randn(batch, steps, top_k, generator=generator)
    hidden_states = torch.randn(batch, steps, hidden, generator=generator)
    hidden_proj = torch.randn(batch, steps, rank, generator=generator) * 0.1
    pred = torch.randn(VOCAB, rank, generator=generator) * 0.1
    succ = torch.randn(VOCAB, rank, generator=generator) * 0.1
    anchors = torch.tensor([1, 4, 7])
    ours = _score_edges(pred, succ, candidate_ids, unary, hidden_proj, anchors, top_k)
    reference = reference_score_edges(pred, succ, candidate_ids, unary, hidden_proj,
                                      anchors, top_k)
    assert ours.shape == (batch, steps, top_k, top_k)
    max_abs = (ours - reference).abs().max().item()
    assert max_abs < 1e-6, f"边分与参考公式差 {max_abs:.3e}"


def test_score_edges_is_context_dependent():
    """同一对 (前驱, 当前候选)，换 hidden 就换分数；hidden=0 时退化成 unary。"""
    candidate_ids = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    unary = torch.tensor([[[0.5, 0.2, -0.1], [0.3, 0.4, 0.1]]])
    # 码本按 `id % 3` 落到三个 rank 上（否则候选 3/4/5 与 rank 0/1/2 的 one-hot 永远不重叠，
    # 构造不出"同一对 (prev,cand) 换 hidden 就换分"的例子）
    pred = torch.nn.functional.one_hot(torch.arange(VOCAB) % 3, 3).float()
    succ = pred * 2.0
    anchors = torch.tensor([9])
    hidden_a = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    hidden_b = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    score_a = _score_edges(pred, succ, candidate_ids, unary, hidden_a, anchors, TOP_K)
    score_b = _score_edges(pred, succ, candidate_ids, unary, hidden_b, anchors, TOP_K)
    assert not torch.allclose(score_a, score_b)
    # 第一步的 k 个 p 共用 anchor：三行必须完全相同
    assert torch.equal(score_a[:, 0, 0], score_a[:, 0, 1])
    assert torch.equal(score_a[:, 0, 0], score_a[:, 0, 2])
    # hidden=0 → 双线性项为 0，只剩 unary（沿 p 广播）
    zero_hidden = torch.zeros_like(hidden_a)
    score_zero = _score_edges(pred, succ, candidate_ids, unary, zero_hidden, anchors,
                              TOP_K)
    assert torch.allclose(score_zero[:, :, :, :], unary[:, :, None, :].expand(-1, -1, 3, -1))


@requires_cuda
def test_upstream_grouped_conv_and_selector_match_ours():
    """**上游真实模块**（`DFlashGroupedConv` / `CandidateSelector`）与我们的实现对照。"""
    generator = torch.Generator().manual_seed(3)
    hidden_size, taps, group_size, block_size, rank = 32, 3, 8, 4, 3
    from minivllm.models.qwen3_dflash2 import CandidateSelector, DFlashGroupedConv

    mine_conv = DFlashGroupedConv(hidden_size, taps, group_size, block_size)
    upstream_conv = make_upstream_conv(hidden_size=hidden_size, taps=taps,
                                       group_size=group_size, block_size=block_size)
    # 两边的参数都是 `torch.empty` 建的：先给 ours 写一份确定值，再拷给上游。**不能**直接
    # 拿未初始化的 ours 去 copy（那会把 NaN/垃圾写进对照，测试会随运行顺序随机失败）。
    with torch.no_grad():
        mine_conv.base_kernel.copy_(
            torch.randn(2, taps, hidden_size, generator=generator) * 0.1)
        mine_conv.kernel_projection.weight.copy_(
            torch.randn(2 * taps * (hidden_size // group_size), hidden_size,
                        generator=generator) * 0.1)
        upstream_conv.base_kernel.copy_(mine_conv.base_kernel)
        upstream_conv.kernel_projection.weight.copy_(mine_conv.kernel_projection.weight)
    states = torch.randn(8, hidden_size, generator=generator)
    with torch.no_grad():
        mine_prepare, mine_coeff = mine_conv.prepare(states)
        up_prepare, up_coeff = upstream_conv.prepare(states)
        assert torch.allclose(mine_coeff, up_coeff, atol=1e-6)
        assert torch.allclose(mine_prepare, up_prepare, atol=1e-6)
        post = torch.randn(8, hidden_size, generator=generator)
        assert torch.allclose(mine_conv.finish(post, mine_coeff),
                              upstream_conv.finish(post, up_coeff), atol=1e-6)

    mine_selector = CandidateSelector(hidden_size, VOCAB, rank, TOP_K)
    upstream_selector = make_upstream_selector(hidden_size=hidden_size,
                                               vocab_size=VOCAB, rank=rank,
                                               top_k=TOP_K)
    with torch.no_grad():
        mine_selector.predecessor_codebook.copy_(
            torch.randn(VOCAB, rank, generator=generator) * 0.2)
        mine_selector.successor_codebook.copy_(
            torch.randn(VOCAB, rank, generator=generator) * 0.2)
        mine_selector.hidden_projection.weight.copy_(
            torch.randn(rank, hidden_size, generator=generator) * 0.2)
        upstream_selector.predecessor_codebook.copy_(
            mine_selector.predecessor_codebook)
        upstream_selector.successor_codebook.copy_(mine_selector.successor_codebook)
        upstream_selector.hidden_projection.weight.copy_(
            mine_selector.hidden_projection.weight)
    candidate_ids = torch.stack([
        torch.randperm(VOCAB, generator=generator)[:TOP_K]
        for _ in range(2 * K)
    ]).view(2, K, TOP_K)
    unary = torch.randn(2, K, TOP_K, generator=generator)
    hidden = torch.randn(2, K, hidden_size, generator=generator)
    anchors = torch.tensor([1, 5])
    with torch.no_grad():
        mine_scores = mine_selector(candidate_ids, unary, hidden, anchors)
        up_scores = upstream_selector(candidate_ids, unary, hidden, anchors)
    assert torch.allclose(mine_scores, up_scores, atol=1e-6)


# ---------------- B. 第一步选择影响第二步 ----------------

def _two_step_case(first_choice: int):
    """手写 `[1,2,3,3]` 的边分：第一步选 `first_choice` 列，第二步的最优候选跟着它走。

    候选：第 0 步 `[1,2,3]`、第 1 步 `[4,5,6]`。第 1 步的 p 行里，**只有与第一步选中列
    同号的那一行**有一个明显更高的分数——于是"第一步 0/1/2 → 第二步 0/1/2"。
    """
    candidate_ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]], device=DEVICE)
    scores = torch.zeros(1, 2, TOP_K, TOP_K, device=DEVICE)
    # 第一步：三行（p）完全相同，只有"当前候选"维不同
    step0 = torch.tensor([0.0, 1.0, 0.5], device=DEVICE)
    scores[0, 0, :, :] = step0
    # 第二步：第 p 行把第 p 列抬高
    for p in range(TOP_K):
        scores[0, 1, p, p] = 2.0
    # 让指定第一步选择成为最优（临时改 step0 的分数）
    order = [0.0, 1.0, 0.5]
    order[first_choice] = 2.0
    scores[0, 0, :, :] = torch.tensor(order, device=DEVICE)
    return candidate_ids, scores


@requires_cuda
@pytest.mark.parametrize("first_choice", [0, 1, 2])
def test_first_choice_decides_the_second_step(first_choice):
    """逐步比较：生产内核 vs 显式参考 vs 上游内核。"""
    speculator, _ = make_speculator(k=2, draft_sample_method="probabilistic",
                                    selector_top_k=TOP_K)
    candidate_ids, scores = _two_step_case(first_choice)
    set_walk_inputs(speculator, anchors=[3], temperature=0.0)
    speculator._sample_path(candidate_ids, scores, 1)
    ours = speculator.draft_tokens[0].clone()
    reference_tokens, reference_scores = reference_greedy_walk(scores, candidate_ids)
    assert torch.equal(ours, reference_tokens[0])
    assert torch.allclose(speculator._selector_scores[0], reference_scores[0])

    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        _selector_walk_kernel)

    up_tokens = torch.zeros(1, 2, dtype=torch.int64, device=DEVICE)
    up_realized = torch.zeros(1, 2, TOP_K, dtype=torch.float32, device=DEVICE)
    _selector_walk_kernel[(1,)](
        scores.contiguous(), candidate_ids.contiguous(), speculator.sample_pos,
        speculator.sample_idx_mapping, speculator.temperature, speculator.seeds,
        up_tokens, up_realized,
        num_steps=2, top_k=TOP_K, BLOCK_K=4, SAMPLE_PROBABILISTIC=False,
        USE_FP64=False, num_warps=1)
    assert torch.equal(ours, up_tokens[0])
    assert torch.equal(speculator._selector_scores[0], up_realized[0])
    # 第二步真的跟着第一步走：第一步列 first_choice 选中的 token 是 first_choice+1，
    # 第二步对应的 p=first_choice 行把第 first_choice 列抬高 → token 4+first_choice。
    assert ours[0].item() == first_choice + 1
    assert ours[1].item() == 4 + first_choice


# ---------------- C/D. 与上游内核逐值差分（B 重排 / slot / 补齐） ----------------

@requires_cuda
@pytest.mark.parametrize("k", [1, 2, 3, 5])
@pytest.mark.parametrize("top_k", [1, 3])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_walk_matches_the_upstream_kernel(k, top_k, temperature):
    """token 与 realized scores 都与上游内核逐位相等（含 K=1 / top-k=1 边界）。"""
    speculator, _ = make_speculator(k=k, draft_sample_method="probabilistic",
                                    selector_top_k=top_k, max_num_seqs=4)
    candidate_ids, scores = walk_case(batch=3, k=k, top_k=top_k, seed=k * 10 + top_k)
    # slot 重排：请求 0/1/2 → slot 2/0/1（验证持久 slot 映射）
    slots = [2, 0, 1]
    set_walk_inputs(speculator, anchors=[1, 4, 7], temperature=temperature, seeds=11,
                    slot_ids=slots)
    speculator._sample_path(candidate_ids, scores, 3)
    ours_tokens = speculator.draft_tokens[:3].clone()
    ours_realized = speculator._selector_scores[:3].clone()

    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        _selector_walk_kernel)

    up_tokens = torch.zeros(3, k, dtype=torch.int64, device=DEVICE)
    up_realized = torch.zeros(3, k, top_k, dtype=torch.float32, device=DEVICE)
    _selector_walk_kernel[(3,)](
        scores.contiguous(), candidate_ids.contiguous(), speculator.sample_pos,
        speculator.sample_idx_mapping, speculator.temperature, speculator.seeds,
        up_tokens, up_realized,
        num_steps=k, top_k=top_k, BLOCK_K=max(1, 1 << (top_k - 1).bit_length()),
        SAMPLE_PROBABILISTIC=True, USE_FP64=False, num_warps=1)
    assert torch.equal(ours_tokens, up_tokens)
    assert torch.equal(ours_realized, up_realized)


@requires_cuda
def test_padded_rows_write_nothing():
    """`sample_idx_mapping=-1` 的行（图重放的补齐）不能写 token 或分数。"""
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic")
    candidate_ids, scores = walk_case(batch=2, k=3, seed=7)
    set_walk_inputs(speculator, anchors=[1, 2], temperature=0.0, slot_ids=[0, -1])
    speculator.draft_tokens.fill_(777)
    speculator._selector_scores.fill_(123.0)
    speculator._sample_path(candidate_ids, scores, 2)
    # 请求 0 正常写；请求 1（slot=-1）保持哨兵
    assert torch.all(speculator.draft_tokens[0] != 777)
    assert torch.all(speculator.draft_tokens[1] == 777)
    assert torch.all(speculator._selector_scores[1] == 123.0)


# ---------------- E. 噪声按 token id 取 ----------------

@requires_cuda
def test_noise_is_keyed_by_token_ids_not_columns():
    """同一组 token 换列排布后，选出来的 token 不变（噪声按 id 抽，不按列号抽）。

    两组的候选集合完全相同、只是列顺序不同；per-token 分数与边分也按 token 对齐。
    如果噪声按列号取，换列之后第一步就可能选到另一个 token。
    """
    speculator, _ = make_speculator(k=2, draft_sample_method="probabilistic")
    run_a_tokens = torch.tensor([[[1, 2, 3], [4, 5, 6]]], device=DEVICE)
    run_b_tokens = torch.tensor([[[3, 1, 2], [6, 4, 5]]], device=DEVICE)
    # 第一步按 token 的分数：token1=0.2、token2=0.7、token3=0.1
    step0_by_token = {1: 0.2, 2: 0.7, 3: 0.1}
    # 第二步：只有 (prev=2, cur=5) 这条边得 50（远大于 Gumbel 噪声，保证它一定赢）
    step1_by_edge = {(2, 5): 50.0}

    def build(tokens):
        scores = torch.zeros(1, 2, TOP_K, TOP_K, device=DEVICE)
        for column, token in enumerate(tokens[0, 0].tolist()):
            scores[0, 0, :, column] = step0_by_token[token]
        for p, prev in enumerate(tokens[0, 0].tolist()):
            for c, cur in enumerate(tokens[0, 1].tolist()):
                if (prev, cur) in step1_by_edge:
                    scores[0, 1, p, c] = step1_by_edge[(prev, cur)]
        return scores

    # 多个种子都要求"两组选出的 token 序列完全相同"：若噪声按列号取，换列之后至少会有
    # 一些种子选出别的 token（单种子可能碰巧相同，所以这里扫 20 个）。
    for seed in range(20):
        set_walk_inputs(speculator, anchors=[9], temperature=0.9, seeds=seed)
        speculator._sample_path(run_a_tokens, build(run_a_tokens), 1)
        tokens_a = speculator.draft_tokens[0].clone()
        set_walk_inputs(speculator, anchors=[9], temperature=0.9, seeds=seed)
        speculator._sample_path(run_b_tokens, build(run_b_tokens), 1)
        tokens_b = speculator.draft_tokens[0].clone()
        assert torch.equal(tokens_a, tokens_b), (seed, tokens_a, tokens_b)


@requires_cuda
def test_edge_scores_select_the_successor():
    """条件 walk 的边语义：第一步选中 token 2 后，第二步只能沿 (prev=2) 那条边取最高分。

    与上一条的区别：这里把第一步的分数抬高到噪声不可能翻盘，于是"选 2 → 第二步选 5"
    是确定性的（验证 `previous` 这个列号真的被喂进第二步的 `[k_prev]` 维）。
    """
    speculator, _ = make_speculator(k=2, draft_sample_method="probabilistic")
    candidate_ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]], device=DEVICE)
    scores = torch.zeros(1, 2, TOP_K, TOP_K, device=DEVICE)
    scores[0, 0, :, :] = torch.tensor([0.0, 50.0, 0.0], device=DEVICE)  # token 2（第 1 列）
    scores[0, 1, 1, 1] = 50.0                            # 上一列=1（token 2）→ token 5
    set_walk_inputs(speculator, anchors=[9], temperature=0.9, seeds=3)
    speculator._sample_path(candidate_ids, scores, 1)
    assert speculator.draft_tokens[0].tolist() == [2, 5]


# ---------------- 提议者从 checkpoint 取 selector_top_k ----------------

def test_speculator_reads_selector_top_k_from_checkpoint():
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic",
                                    selector_top_k=4)
    assert speculator.selector_top_k == 4
    assert speculator._selector_scores.shape == (speculator.max_num_reqs, 3, 4)
    assert speculator._cached_candidate_ids.shape == (speculator.max_num_reqs, 3, 4)
    # `_anchor_indices` 的步长 = 每请求 query 行数（1 + K）
    assert speculator._anchor_indices[:2].tolist() == [0, 4]
