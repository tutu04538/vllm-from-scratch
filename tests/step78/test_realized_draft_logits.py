"""step78：realized draft logits（实际路径的 q）与验证器读到的那一份。

需求 078 §3 的硬要求：

    q 是 fp32、初值 `-inf`；只写实际路径的 k 个候选列；**旧候选列必须被清成 `-inf`**
    （slot 复用、候选集换了以后，旧列不能变成"幽灵候选"）；walk 与 verifier 看到同一份数。

这里把 `_cache_draft_logits_kernel` 单拎出来跑两轮（同 slot、不同候选集），既与**上游内核**
逐位比，也与"实际路径"逐值比；再从 q 抽 2000 次验证经验分布。
"""

import pytest
import torch

from dflash2_helpers import (DEVICE, make_speculator, set_walk_inputs, walk_case)
from minivllm.testing.torch_dflash2 import reference_greedy_walk

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="候选 walk / 缓存是 Triton 内核（上游同款边界），只在 CUDA 上可执行")

VOCAB = 11
TOP_K = 3


def _candidates(*rows):
    """把若干行候选（list）拼成 `[1, K, k]`。"""
    return torch.tensor([list(row) for row in rows], dtype=torch.int64,
                        device=DEVICE).unsqueeze(0)


def _run_round(speculator, candidate_ids, *, slot=0, temperature=0.0, seed=0):
    """跑一轮 walk + 落 q，返回 `(tokens, realized_scores, q[slot])`。"""
    set_walk_inputs(speculator, anchors=[3], temperature=temperature, seeds=seed,
                    slot_ids=[slot])
    scores = torch.randn(candidate_ids.shape[0], candidate_ids.shape[1], TOP_K, TOP_K,
                         device=DEVICE)
    speculator._sample_path(candidate_ids, scores, candidate_ids.shape[0])
    realized = speculator._selector_scores[:candidate_ids.shape[0]].clone()
    if speculator.draft_logits is not None:
        speculator._cache_draft_logits(candidate_ids,
                                       candidate_ids.shape[0] * candidate_ids.shape[1])
    return (speculator.draft_tokens[:candidate_ids.shape[0]].clone(), realized,
            None if speculator.draft_logits is None else
            speculator.draft_logits[slot].clone())


def test_draft_logits_spec_is_fp32_and_minus_inf():
    """概率草稿：fp32 + 全 `-inf` 初值；贪心草稿：根本没有 q（点质量）。"""
    probabilistic, _ = make_speculator(k=3, draft_sample_method="probabilistic")
    assert probabilistic.draft_logits is not None
    assert probabilistic.draft_logits.dtype == torch.float32
    assert torch.all(probabilistic.draft_logits == float("-inf"))
    greedy, _ = make_speculator(k=3, draft_sample_method="greedy")
    assert greedy.draft_logits is None


@requires_cuda
def test_cache_writes_only_the_k_candidates():
    """q 的每一行恰好 k 个有限列，且**列值就是 realized scores**。"""
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic")
    candidate_ids = _candidates([0, 5, 9], [2, 4, 8], [1, 6, 7])
    _, realized, q = _run_round(speculator, candidate_ids)
    assert q is not None
    for step in range(3):
        finite = torch.isfinite(q[step])
        assert int(finite.sum()) == TOP_K
        assert set(q[step][finite].tolist()) == set(realized[0, step].tolist())
        # 有限列正好是这一轮写进去的候选
        expected = set(candidate_ids[0, step].tolist())
        assert set(torch.nonzero(finite).flatten().tolist()) == expected
        # 非候选列必须是 -inf（不能是 0）
        assert torch.all(q[step][~finite] == float("-inf"))
    # softmax 是合法分布
    probs = torch.softmax(q, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3, device=DEVICE), atol=1e-6)


@requires_cuda
def test_disjoint_candidate_sets_clear_the_old_columns():
    """新旧候选集**完全不相交**：旧列恢复 `-inf`（需求 078 §3）。"""
    speculator, _ = make_speculator(k=1, draft_sample_method="probabilistic")
    first = _candidates([0, 1, 2])
    _, realized_first, q_first = _run_round(speculator, first, slot=0, seed=1)
    assert set(torch.nonzero(torch.isfinite(q_first[0])).flatten().tolist()) == {0, 1, 2}

    second = _candidates([3, 4, 5])
    _, realized_second, q_second = _run_round(speculator, second, slot=0, seed=2)
    finite_second = torch.isfinite(q_second[0])
    assert set(torch.nonzero(finite_second).flatten().tolist()) == {3, 4, 5}
    # 旧列 0/1/2 必须回到 -inf
    assert torch.all(q_second[0][:3] == float("-inf")), q_second[0]
    # 新列的值就是第二轮的实际路径分数
    assert set(q_second[0][finite_second].tolist()) == set(realized_second[0, 0].tolist())
    # `_cached_candidate_ids` 也跟着换成新候选（下一轮清理的依据）
    assert set(speculator._cached_candidate_ids[0, 0].tolist()) == {3, 4, 5}


@requires_cuda
def test_overlapping_candidate_sets_update_shared_columns():
    """新旧候选集**有交集**：交集列必须更新成新分数（先清后写），不能被旧值粘住。"""
    speculator, _ = make_speculator(k=1, draft_sample_method="probabilistic")
    _, realized_first, q_first = _run_round(speculator, _candidates([0, 1, 2]), slot=0,
                                            seed=3)
    old_value_of_2 = q_first[0, 2].item()

    _, realized_second, q_second = _run_round(speculator, _candidates([2, 4, 6]), slot=0,
                                              seed=4)
    finite = torch.isfinite(q_second[0])
    assert set(torch.nonzero(finite).flatten().tolist()) == {2, 4, 6}
    assert torch.all(q_second[0][[0, 1]] == float("-inf"))
    # 交集列 2 是新分数（两轮 realized 不同，才能证明"更新"而不是"碰巧相同"）
    new_value_of_2 = q_second[0, 2].item()
    assert new_value_of_2 != old_value_of_2
    assert new_value_of_2 in set(realized_second[0, 0].tolist())


@requires_cuda
def test_cache_kernel_matches_the_upstream_kernel_on_slot_reuse():
    """同一 slot 连跑两轮、候选集换掉：与上游 `_cache_draft_logits_kernel` 逐位相等。"""
    speculator, _ = make_speculator(k=2, draft_sample_method="probabilistic")
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        _cache_draft_logits_kernel)

    up_logits = torch.full_like(speculator.draft_logits, float("-inf"))
    up_cached = torch.zeros_like(speculator._cached_candidate_ids)

    for candidate_ids in (_candidates([0, 1, 2], [3, 4, 5]),
                          _candidates([2, 6, 7], [8, 9, 10])):
        # 同一组候选/分数喂给两套实现
        set_walk_inputs(speculator, anchors=[3], temperature=0.0, seeds=5, slot_ids=[0])
        scores = torch.randn(candidate_ids.shape[0], candidate_ids.shape[1], TOP_K,
                             TOP_K, device=DEVICE)
        speculator._sample_path(candidate_ids, scores, candidate_ids.shape[0])
        num_sample = candidate_ids.shape[0] * candidate_ids.shape[1]
        speculator._cache_draft_logits(candidate_ids, num_sample)
        _cache_draft_logits_kernel[(num_sample,)](
            up_logits, up_cached, candidate_ids,
            speculator._selector_scores[:candidate_ids.shape[0]],
            speculator.sample_idx_mapping,
            up_logits.stride(0), up_logits.stride(1),
            num_steps=candidate_ids.shape[1], top_k=TOP_K, BLOCK_K=4, num_warps=1)
        assert torch.equal(speculator.draft_logits, up_logits)
        assert torch.equal(speculator._cached_candidate_ids, up_cached)


@requires_cuda
def test_generate_draft_ties_every_layer_together(monkeypatch):
    """`_generate_draft` 全流程：候选 → 边分 → walk → 落 q，q 必须等于实际路径。

    主干前向与候选 selector 都被替换成手写值（`_run_model` 返回固定 hidden），于是
    "第几步选了什么、q 该是什么"可以独立算出来。
    """
    speculator, _ = make_speculator(k=2, draft_sample_method="probabilistic")
    candidate_ids = _candidates([0, 3, 7], [1, 5, 9])
    unary = torch.tensor([[0.5, 0.1, -0.2], [0.4, 0.3, 0.2]], device=DEVICE)
    scores = torch.zeros(1, 2, TOP_K, TOP_K, device=DEVICE)
    # 第一步：按 unary + 一个让 3 号候选赢的偏移（通过 p=0 行也一起设）
    scores[0, 0, :, :] = torch.tensor([0.0, 1.0, -0.5], device=DEVICE)
    # 第二步：上一步选列 1 → 只有 p=1 行给 9 号候选高分
    scores[0, 1, :, :] = 0.0
    scores[0, 1, 1, 2] = 3.0

    num_steps = speculator.num_speculative_steps
    num_tokens = 1 + num_steps  # DFlash 的 1+K query 行

    monkeypatch.setattr(
        speculator, "_run_model",
        lambda num_tokens, attn, slots, mode: torch.zeros(
            num_tokens, speculator.hidden_size, device=DEVICE))
    monkeypatch.setattr(
        speculator.model, "compute_candidates",
        lambda hidden: (candidate_ids.view(-1, TOP_K), unary))
    monkeypatch.setattr(
        speculator.model.model.candidate_selector, "forward",
        lambda candidates, logits, hidden, anchors: scores)

    set_walk_inputs(speculator, anchors=[4], temperature=0.0, seeds=0, slot_ids=[0])
    speculator._generate_draft(1, num_tokens, None, None)

    reference_tokens, reference_scores = reference_greedy_walk(scores, candidate_ids)
    assert torch.equal(speculator.draft_tokens[0], reference_tokens[0])
    q = speculator.draft_logits[0]
    for step in range(num_steps):
        finite = torch.isfinite(q[step])
        assert set(torch.nonzero(finite).flatten().tolist()) == \
            set(candidate_ids[0, step].tolist())
        assert set(q[step][finite].tolist()) == set(reference_scores[0, step].tolist())


@requires_cuda
def test_empirical_draws_match_the_cached_q():
    """从 q 抽 2000 个种子：经验分布 vs `softmax(q / temperature)`（TV < 0.1）。

    温度 2.0 同时证明"缓存的 q 是温度之前的那一份"——若缓存里存的是 `scores/temp`，
    经验分布会与 `softmax(q)` 对上而不是 `softmax(q/temp)`。
    """
    speculator, _ = make_speculator(k=1, draft_sample_method="probabilistic",
                                    selector_top_k=TOP_K)
    candidate_ids = _candidates([0, 5, 9])
    # K=1 时第一步的 k 个 p 行都用 anchor，所以三行分数必须相同：[1,1,k,k]
    scores = torch.tensor([1.0, 0.2, -0.7], device=DEVICE).view(1, 1, 1, TOP_K).expand(
        1, 1, TOP_K, TOP_K).contiguous()
    temperature = 2.0
    counts = torch.zeros(TOP_K, dtype=torch.float64, device=DEVICE)
    num_trials = 2000
    for seed in range(num_trials):
        set_walk_inputs(speculator, anchors=[3], temperature=temperature, seeds=seed,
                        slot_ids=[0])
        speculator._sample_path(candidate_ids, scores, 1)
        token = speculator.draft_tokens[0, 0].item()
        column = int((candidate_ids[0, 0] == token).nonzero().flatten()[0])
        counts[column] += 1
    speculator._cache_draft_logits(candidate_ids, 1)
    q = speculator.draft_logits[0, 0]
    # 只取候选那三列（q 的其余列是 -inf，softmax 的全词表质量本来就落在候选上）
    expected = torch.softmax(q / temperature, dim=-1)[candidate_ids[0, 0]]
    empirical = counts / counts.sum()
    tv = 0.5 * (empirical - expected).abs().sum().item()
    assert tv < 0.1, f"经验分布与 softmax(q/temp) 的 TV={tv:.3f} 太大"
    # q 里存的是原始分数（不是除以温度之后的）
    assert q[0].item() == pytest.approx(1.0, abs=1e-6)
    assert q[5].item() == pytest.approx(0.2, abs=1e-6)
    assert q[9].item() == pytest.approx(-0.7, abs=1e-6)


@requires_cuda
def test_greedy_walk_uses_argmax_and_keeps_no_q():
    """greedy：q 是点质量（`draft_logits=None`），token 就是 realized 分数里的 argmax。"""
    speculator, _ = make_speculator(k=3, draft_sample_method="greedy")
    candidate_ids, scores = walk_case(batch=2, k=3, seed=9)
    set_walk_inputs(speculator, anchors=[1, 2], temperature=0.0, seeds=0)
    speculator._sample_path(candidate_ids, scores, 2)
    reference_tokens, _ = reference_greedy_walk(scores, candidate_ids)
    assert speculator.draft_logits is None
    assert torch.equal(speculator.draft_tokens[:2], reference_tokens)
