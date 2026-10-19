"""step77：DSpark 的**条件采样**与"实际 proposal 分布"。

四条主线（对应用需求 077 §4 的验收清单）：

    A. **条件分布随前一个 token 变化**：同一份 base logits，anchor 不同 → 第一步选出的 token 不同
       → 第二步的 q 不同。用一组手写数字（不是 tiny 权重）钉死，避免"恰好相同"。
    B. **缓存的 q == 实际用的 q**：把 `draft_logits` 里那一份与冻的参考实现逐值比，并且
       **沿实际采样出来的路径**复算（bias 来自上一步真选中的 token，不是 anchor）。
       概率草稿在温度 0 与 0.7 下都比一遍：温度 0.7 同时证明"缓存的是温度之前的那一份"。
    C. **缩减词表 / top-k**：q 是 target 宽度、非候选列恒 `-inf`、重复 batch 不残留旧值、
       非零偏移的 d2t 真的被加上；top-k 取满词表时与不开 top-k 逐值相同。
    D. **采样确实从 q 抽**：2000 个种子的经验边际 vs `softmax(q)`（TV 阈值 0.1）。

置信度头的落盘（79 关的原料）与 greedy/probabilistic、图/eager 的对照也在这里。
"""

import pytest
import torch
import torch.nn.functional as F

from minivllm import SamplingParams
from minivllm.testing.torch_dspark import reference_sequential

from dspark_helpers import (DEVICE, make_engine, make_speculator, run_prompts,  # noqa: E402
                            set_sampling)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="概率草稿的 Gumbel 采样是 Triton 内核（上游同款边界），只在 CUDA 上可执行")

VOCAB = 11


def _markov_weights(speculator):
    head = speculator.model.model.markov_head
    return head.markov_w1.weight, head.markov_w2.weight


def _set_markov_weights(speculator, w1_rows: dict, w2_rows: dict) -> None:
    """按"行 → 值"改写 Markov 头（测试手写分布，避免依赖 tiny 随机权重）。"""
    w1, w2 = _markov_weights(speculator)
    with torch.no_grad():
        w1.zero_()
        w2.zero_()
        for token, value in w1_rows.items():
            w1[token] = torch.tensor(value, dtype=w1.dtype, device=w1.device)
        for token, value in w2_rows.items():
            w2[token] = torch.tensor(value, dtype=w2.dtype, device=w2.device)


def _base_logits(num_reqs, k, values=None):
    base = torch.zeros(num_reqs, k, VOCAB, device=DEVICE)
    if values is not None:
        for (req, step), row in values.items():
            row_t = torch.tensor(row, dtype=base.dtype, device=DEVICE)
            base[req, step, : row_t.numel()] = row_t   # 尾部按 0 补齐
    return base


def _draft_ids_of(model, target_ids: torch.Tensor) -> torch.Tensor:
    """target id → draft id（按 `d2t` 的映射求逆；全词表时是恒等）。"""
    d2t = model.draft_id_to_target_id
    if d2t is None:
        return target_ids
    draft = torch.arange(d2t.shape[0], device=d2t.device)
    table = draft + d2t
    match = (target_ids.unsqueeze(-1) == table.unsqueeze(0)
             ).to(torch.int64).argmax(dim=-1)
    assert torch.equal(table[match], target_ids), "采样到了 d2t 之外的目标 id"
    return match


# ---------------- A. 条件分布依赖前一个 token ----------------

@requires_cuda
def test_conditional_distribution_depends_on_the_previous_token():
    """同一份 base logits：anchor 换成另一个 ⇒ 后一位置的条件分布跟着变。

    手写数字（K=3、词表 11、r=3、logit_scale=1）：

        markov_w1[1] = [0,1,0]   → anchor=1 时偏置 = w2[:,1] = 在 token 1 上加 3
        markov_w1[2] = [1,0,0]   → anchor=2 时偏置 = w2[:,0] = 在 token 0 上加 3
        其余行全 0（所以 prev=0 的偏置恰好是 0，第 1 步就退化成 base logits）

    于是：
        运行 A（anchor=2）  第 0 步选 0，第 1 步 q = [2, 1, 0, …] → 选 0
        运行 B（anchor=1）  第 0 步选 1，第 1 步 q = [2, 4, 0, …] → 选 1
    两条路径第 1 步的 **base logits 完全相同**，q 却不同 —— 这就是块内依赖。
    """
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic",
                                    logit_scale=1.0)
    _set_markov_weights(
        speculator,
        w1_rows={1: [0.0, 1.0, 0.0], 2: [1.0, 0.0, 0.0]},
        w2_rows={0: [3.0, 0.0, 0.0], 1: [0.0, 3.0, 0.0]},
    )
    base = _base_logits(1, 3, values={(0, 1): [2.0, 1.0, 0.0]})
    restore = None
    results = {}
    try:
        for name, anchor in (("A", 2), ("B", 1)):
            n = set_sampling(speculator, [anchor], temperature=0.0)
            restore = speculator.model.compute_draft_logits
            flat = base.reshape(-1, VOCAB)
            speculator.model.compute_draft_logits = lambda hidden, _flat=flat: _flat
            speculator._sample_sequential(1, torch.zeros(n, 32, device=DEVICE))
            results[name] = (
                speculator.draft_tokens[0].clone(),
                speculator.draft_logits[0].clone(),
            )
            speculator.model.compute_draft_logits = restore
    finally:
        if restore is not None:
            speculator.model.compute_draft_logits = restore

    tokens_a, q_a = results["A"]
    tokens_b, q_b = results["B"]
    # 第 0 步就被 anchor 的偏置分开了
    assert tokens_a[0].item() == 0
    assert tokens_b[0].item() == 1
    # 第 1 步：base 相同、q 不同（差在 token 0/1 那一列）
    want_a = torch.tensor([2.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=DEVICE)
    want_b = torch.tensor([2.0, 4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=DEVICE)
    # 存下来的是 [K, V]（请求维已经切掉），所以 `[1]` 就是第 1 步那一行
    assert torch.allclose(q_a[1], want_a, atol=1e-6), q_a[1]
    assert torch.allclose(q_b[1], want_b, atol=1e-6), q_b[1]
    assert not torch.allclose(q_a[1], q_b[1]), "前一 token 变了，后一位置的 q 却没变"
    assert tokens_a[1].item() == 0 and tokens_b[1].item() == 1

    # 与冻的参考实现一致（沿各自实际选中的路径复算）
    w1, w2 = _markov_weights(speculator)
    for name, anchor, tokens, q in (("A", 2, tokens_a, q_a), ("B", 1, tokens_b, q_b)):
        ref_tokens, ref_q = reference_sequential(
            base, torch.tensor([anchor], device=DEVICE), w1, w2,
            choose=lambda logits, step, _t=tokens: _t[step].unsqueeze(0),
            logit_scale=1.0)
        assert torch.equal(ref_tokens[0], tokens), name
        assert torch.allclose(ref_q[0], q, atol=1e-6), name


# ---------------- B. 缓存的 q == 实际采样用的 q ----------------

@requires_cuda
@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("anchor", [True, False])
def test_cached_q_matches_the_actual_sampled_path(temperature, anchor):
    """概率草稿：`draft_logits[req, step]` 必须等于该步**实际用的** logits（温度之前）。"""
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic",
                                    anchor=anchor, logit_scale=1.7)
    n = set_sampling(speculator, [1, 4], temperature=temperature, seeds=7)
    hidden = torch.randn(n, 32, device=DEVICE)
    base = speculator.model.compute_draft_logits(hidden).view(2, 3, -1)
    speculator._sample_sequential(2, hidden)

    tokens = speculator.draft_tokens[:2].clone()
    q = speculator.draft_logits[:2].clone()
    w1, w2 = _markov_weights(speculator)
    ref_tokens, ref_q = reference_sequential(
        base, torch.tensor([1, 4], device=DEVICE), w1, w2,
        choose=lambda logits, step, _t=tokens: _draft_ids_of(
            speculator.model, _t[:, step]),
        logit_scale=speculator.model.logits_processor.scale)
    assert torch.equal(ref_tokens, tokens), "沿实际路径复算出的 token 与生产路径不同"
    assert torch.allclose(ref_q, q, atol=1e-6), (
        "缓存的 q 与'实际采样用的条件分布'不一致 —— 拒绝采样会用到错的 q")
    # q 里必须含 Markov 偏置（否则就是"只缓存了 base logits"）
    assert not torch.allclose(q, base), "q 与 base logits 相同：偏置没有进 q"


@requires_cuda
def test_probabilistic_is_stochastic_but_greedy_is_not():
    """温度 > 0 时不同 seed 走不同路径；温度 0 时两条路径都应是 argmax。"""
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic")
    n = set_sampling(speculator, [1], temperature=1.0, seeds=1)
    hidden = torch.randn(n, 32, device=DEVICE)
    paths = []
    for seed in (1, 2, 3):
        set_sampling(speculator, [1], temperature=1.0, seeds=seed)
        speculator._sample_sequential(1, hidden)
        paths.append(speculator.draft_tokens[0].clone())
    assert len({tuple(path.tolist()) for path in paths}) >= 2, (
        "三个种子走了完全相同的路径：概率采样没有生效")

    set_sampling(speculator, [1], temperature=0.0, seeds=99)
    speculator._sample_sequential(1, hidden)
    first = speculator.draft_tokens[0].clone()
    set_sampling(speculator, [1], temperature=0.0, seeds=123)
    speculator._sample_sequential(1, hidden)
    assert torch.equal(first, speculator.draft_tokens[0]), "温度 0 应该与种子无关"


# ---------------- C. 缩减词表与 top-k ----------------

@requires_cuda
def test_reduced_vocab_q_is_a_full_target_distribution():
    """缩减词表：q 是 target 宽度，非 d2t 列恒 `-inf`，d2t 偏移真的被加上。"""
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic",
                                    draft_vocab_size=7, d2t_offset=2)
    model = speculator.model
    assert model.draft_id_to_target_id is not None
    offsets = model.draft_id_to_target_id.clone()
    assert torch.equal(offsets, torch.full_like(offsets, 2)), offsets
    # d2t 列 = {2..8}；0/1/9/10 永远不会被提议
    allowed = torch.arange(7, device=DEVICE) + offsets
    forbidden = sorted(set(range(VOCAB)) - set(allowed.tolist()))
    assert forbidden == [0, 1, 9, 10]

    n = set_sampling(speculator, [1, 4], temperature=0.0)
    hidden = torch.randn(n, 32, device=DEVICE)
    speculator._sample_sequential(2, hidden)
    q = speculator.draft_logits[:2].clone()
    tokens = speculator.draft_tokens[:2].clone()
    assert q.shape == (2, 3, VOCAB)
    for column in forbidden:
        assert torch.isinf(q[:, :, column]).all() and (q[:, :, column] < 0).all(), column
    assert q[:, :, list(allowed.tolist())].isfinite().all()
    # 采样出来的 id 一定落在映射后的 target 空间（不是 draft 空间的 0..6）
    assert set(tokens.flatten().tolist()) <= set(allowed.tolist()), tokens
    # q 是完整分布：softmax 在 target 宽度上和为 1
    probs = torch.softmax(q, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-6)

    # 参考：draft 空间的 logits 经同一张 d2t 散射
    base = speculator.model.compute_draft_logits(hidden).view(2, 3, -1)
    w1, w2 = _markov_weights(speculator)
    _, ref_q = reference_sequential(
        base, torch.tensor([1, 4], device=DEVICE), w1, w2,
        choose=lambda logits, step, _t=tokens: _draft_ids_of(model, _t[:, step]),
        logit_scale=model.logits_processor.scale, d2t=model.draft_id_to_target_id,
        target_vocab=VOCAB)
    assert torch.allclose(ref_q, q, atol=1e-6)


@requires_cuda
def test_repeated_batches_never_inherit_stale_logits():
    """同一个 slot 连着跑两批不同的 base logits：q 必须整份被覆盖（不残留上一批的列）。"""
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic",
                                    draft_vocab_size=7, d2t_offset=2)
    n = set_sampling(speculator, [1, 4], temperature=0.0)
    hidden_one = torch.randn(n, 32, device=DEVICE)
    speculator._sample_sequential(2, hidden_one)
    first = speculator.draft_logits[:2].clone()

    hidden_two = torch.randn(n, 32, device=DEVICE)
    speculator._sample_sequential(2, hidden_two)
    second = speculator.draft_logits[:2].clone()

    allowed = (torch.arange(7, device=DEVICE) + 2).tolist()
    forbidden = sorted(set(range(VOCAB)) - set(allowed))
    assert not torch.allclose(first, second), "两批不同的输入给出了同一份 q"
    for column in forbidden:
        assert torch.isinf(second[:, :, column]).all(), column
    # 每个 (req, step) 的 d2t 列都被第二批整体覆盖
    base_two = speculator.model.compute_draft_logits(hidden_two).view(2, 3, -1)
    w1, w2 = _markov_weights(speculator)
    _, ref_q = reference_sequential(
        base_two, torch.tensor([1, 4], device=DEVICE), w1, w2,
        choose=lambda logits, step: _draft_ids_of(
            speculator.model, speculator.draft_tokens[:2][:, step]),
        logit_scale=speculator.model.logits_processor.scale,
        d2t=speculator.model.draft_id_to_target_id, target_vocab=VOCAB)
    assert torch.allclose(ref_q, second, atol=1e-6)


@requires_cuda
def test_topk_truncates_the_q_to_the_candidates():
    """`dspark_draft_topk`：每步只有 k 个有限列，其余 `-inf`；采样 id 一定在候选里。"""
    k_candidates = 3
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic",
                                    draft_topk=k_candidates)
    assert speculator._draft_topk == k_candidates
    n = set_sampling(speculator, [1, 4], temperature=0.0)
    hidden = torch.randn(n, 32, device=DEVICE)
    base = speculator.model.compute_draft_logits(hidden).view(2, 3, -1)
    speculator._sample_sequential(2, hidden)

    q = speculator.draft_logits[:2].clone()
    tokens = speculator.draft_tokens[:2].clone()
    finite = torch.isfinite(q)
    assert finite.sum(-1).tolist() == [[k_candidates] * 3] * 2, finite.sum(-1)
    candidates = base.topk(k_candidates, dim=-1).indices
    for req in range(2):
        for step in range(3):
            assert tokens[req, step].item() in candidates[req, step].tolist()
            assert finite[req, step, candidates[req, step]].all()

    w1, w2 = _markov_weights(speculator)
    _, ref_q = reference_sequential(
        base, torch.tensor([1, 4], device=DEVICE), w1, w2,
        choose=lambda logits, step: tokens[:, step],
        logit_scale=speculator.model.logits_processor.scale,
        topk=k_candidates)
    assert torch.allclose(ref_q, q, atol=1e-6)


@requires_cuda
def test_topk_over_the_full_vocab_equals_no_topk():
    """top-k 取满词表时应与不截断**逐值相同**（候选 = 全部列，等价于额外的 scatter）。"""
    plain, _ = make_speculator(k=3, draft_sample_method="probabilistic")
    full, _ = make_speculator(k=3, draft_sample_method="probabilistic", draft_topk=VOCAB)
    hidden = torch.randn(6, 32, device=DEVICE)
    for speculator in (plain, full):
        set_sampling(speculator, [1, 4], temperature=0.0)
        speculator._sample_sequential(2, hidden)
    assert torch.equal(plain.draft_tokens[:2], full.draft_tokens[:2])
    # q 只允许末位差：两条路的 Markov 偏置一个是 `F.linear`（dense）、一个是 `baddbmm`
    # （gather），浮点末位可能差 1 ulp；分布本身相同（同一组候选、同一组数值）。
    assert torch.allclose(plain.draft_logits[:2], full.draft_logits[:2], atol=1e-6)


# ---------------- D. 采样确实从 q 抽 ----------------

@requires_cuda
def test_gumbel_step_matches_the_softmax_of_its_input():
    """2000 个种子：单步采样的经验分布应逼近 `softmax(q)`（TV < 0.1）。

    这条补上"q 对了"之外的另一半：采样器**消费的**就是传进去的那份 logits。如果它偷偷用
    base logits（丢掉 Markov 偏置），这里立刻偏离。
    """
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic")
    logits = torch.randn(1, VOCAB, device=DEVICE) * 2.0
    expected = torch.softmax(logits[0], dim=-1)
    counts = torch.zeros(VOCAB, dtype=torch.float64, device=DEVICE)
    draws = 2000
    idx_map = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    pos = torch.tensor([10], dtype=torch.int64, device=DEVICE)
    for seed in range(draws):
        speculator.temperature.fill_(1.0)
        speculator.seeds.fill_(seed)
        token = speculator._sample_logits(logits, idx_map, pos, 0)
        counts[token.item()] += 1
    empirical = counts / draws
    tv = 0.5 * (empirical - expected.double()).abs().sum()
    assert tv < 0.1, f"经验分布与 softmax(q) 的 TV={tv:.4f} 太大"


# ---------------- 置信度（79 关的原料） ----------------

@requires_cuda
def test_confidence_head_records_per_position_probs():
    """打开 `enable_adaptive_verification` 时，每枚草稿写一行接受概率估计（本关不用它裁剪）。"""
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic",
                                    confidence_head=True)
    assert speculator.model.model.confidence_head is not None
    speculator.enable_adaptive_verification = True
    n = set_sampling(speculator, [1, 4], temperature=0.0)
    hidden = torch.randn(n, 32, device=DEVICE)
    speculator._sample_sequential(2, hidden)

    probs = speculator.draft_token_confidence_probs[:2].clone()
    assert probs.shape == (2, 3) and probs.dtype == torch.float32
    assert ((probs > 0) & (probs < 1)).all()

    # 参考：sigmoid(proj([hidden, markov_embed]))，markov_embed 取**实际路径**上的前一个 token
    head = speculator.model.model.confidence_head
    w1, _ = _markov_weights(speculator)
    prev = torch.cat([
        torch.tensor([1, 4], device=DEVICE).unsqueeze(1),
        speculator.draft_tokens[:2, :-1],
    ], dim=1)
    sample_hidden = hidden[:n].view(2, 3, -1)
    embeds = w1[prev]
    x = torch.cat([sample_hidden, embeds], dim=-1).reshape(-1, head.proj.in_features)
    want = torch.sigmoid(F.linear(x.float(), head.proj.weight, head.proj.bias)).view(2, 3)
    assert torch.allclose(probs, want, atol=1e-6), (probs, want)

    # 关掉开关时不再写（保持上一轮的旧值是上游语义：消费方只在开关打开时读）
    speculator.enable_adaptive_verification = False
    speculator.draft_token_confidence_probs.zero_()
    set_sampling(speculator, [1, 4], temperature=0.0)
    speculator._sample_sequential(2, hidden)
    assert torch.equal(speculator.draft_token_confidence_probs[:2],
                       torch.zeros(2, 3, device=DEVICE))


# ---------------- 端到端：greedy/probabilistic、图/eager ----------------

@requires_cuda
@pytest.mark.parametrize("anchor", [True, False])
def test_graph_and_eager_produce_the_same_drafts(anchor):
    """图里录的是"主干前向 + K 步串行采样"：逐轮草稿 token 必须与 eager 完全相同。"""
    prompts = [("a", [1, 2, 3, 4]), ("b", [5, 6])]
    rounds = {}
    finals = {}
    for mode in ("none", "full_decode_only"):
        engine, _core, runner = make_engine(
            k=3, draft_sample_method="probabilistic", mode=mode,
            capture_sizes=[6, 12], anchor=anchor)
        try:
            for req_id, prompt in prompts:
                engine.add_request(req_id, list(prompt),
                                   SamplingParams(max_tokens=6, temperature=0.7,
                                                  seed=1234, eos_token_id=99))
            snapshot, final = [], {}
            for _ in range(200):
                if not engine.has_unfinished_requests():
                    break
                for out in engine.step():
                    final[out.request_id] = list(out.token_ids)
                snapshot.append(runner.req_states.draft_tokens[:2].clone())
        finally:
            engine.shutdown()
        rounds[mode] = snapshot
        finals[mode] = final
        if mode == "full_decode_only":
            stats = runner.cudagraph_capture_stats or {}
            assert stats.get("captured", 0) >= 1, stats
            selections = runner.speculator.query_cudagraph_manager.selections
            assert selections, "图管理器没有记录任何一次分派"

    assert finals["none"] == finals["full_decode_only"]
    assert len(rounds["none"]) == len(rounds["full_decode_only"]) > 1
    for index, (eager, graph) in enumerate(zip(rounds["none"], rounds["full_decode_only"])):
        assert torch.equal(eager, graph), f"第 {index} 轮的草稿 token 不同"


@requires_cuda
def test_greedy_and_probabilistic_agree_at_temperature_zero():
    """温度 0 时两条草稿路径都退化成 argmax：最终 token 必须相同（无损性）。"""
    prompts = [("a", [1, 2, 3, 4]), ("b", [5, 6])]
    outputs = {}
    for method in ("greedy", "probabilistic"):
        engine, _core, runner = make_engine(k=3, draft_sample_method=method)
        assert (runner.speculator.draft_logits is not None) == (method == "probabilistic")
        outputs[method] = run_prompts(engine, prompts, max_tokens=6, temperature=0.0)
    assert outputs["greedy"] == outputs["probabilistic"]
