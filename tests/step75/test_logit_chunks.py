"""75 关：**分块 logits**（`_verify_in_chunks` / `_iter_request_chunks` / `get_max_chunk_logits`）。

要解决的问题（075 §1）：验证前要先 `apply_sampling_params()` 把整段 target logits 升到 fp32 再
原地处理，而投机批的行数是 `Σ(1 + K_i)`。15 万词表下**一行就 0.58 MiB**，512 条请求 × 5 行
≈ 1.5 GiB——不设上限，显存随并发线性涨。上限 = `MAX_CHUNK_BYTES // (vocab × 4)`（1 GiB 的 fp32
行数），超了就按**整条请求**切块，逐块验证、逐块释放。

这一关在切块上要钉三件事（075 §3.6/§3.7）：

    C-offset  chunk 内的 `cu_num_logits` 必须重置成**块局部**偏移（`- lo`）；
              而 `draft_logits` 按**常驻 request-state slot** 寻址，是全局的，**不能跟着切**。
    C-order   多块的 `sampled`/`num_sampled`/logprobs 拼回来之后，顺序与返回结构与单块一致。
    C-stale   adaptive verification（79 关）会把 logits 压紧却**不更新** `cu_num_logits_np`
              （保留压紧前的布局）；那些"陈旧的 CPU 前缀和"绝不能再用来挑块边界。本关的保护就是
              `logits.shape[0] <= max_chunk_logits → 单块`（79 关给压紧后的批做了预算上限，
              保证它一定落进这一档），这里用"陈旧 offsets + 行数不超上限"的构造把它钉住。

本文件分两层：

- `_verify_in_chunks` 层（替身采样器）：精确控制块边界、陈旧 offsets、logprobs，比逐值。
- 引擎层（tiny V2）：把 `get_max_chunk_logits` 换成很小的值，跑真引擎，与不切块的同一 workload
  逐 token / 逐 logprob 比——这是"切块不改变交付"的端到端证据。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import pytest

from spec75_helpers import (  # noqa: E402
    DEVICE,
    INT32,
    VOCAB,
    layout_of,
    make_v2_engine,
    requires_cuda,
)

pytestmark = requires_cuda

import minivllm.worker.gpu.spec_decode.rejection_sampler as mv_rs  # noqa: E402
from minivllm.config import SpeculativeConfig  # noqa: E402
from minivllm.worker.gpu.spec_decode.rejection_sampler import (  # noqa: E402
    MAX_CHUNK_BYTES,
    RejectionSampler,
    _iter_request_chunks,
    get_max_chunk_logits,
)

K = 3
MAX_NUM_REQS = 8


# ---------------------------------------------------------------------------
# 一、上限与切块规则（纯 numpy）
# ---------------------------------------------------------------------------


def test_chunk_budget_is_a_fp32_row_budget() -> None:
    """上限口径：`1 GiB / (vocab × 4)` 行——词表越大一次能放的行越少（075 §2）。"""
    assert get_max_chunk_logits(151936) == MAX_CHUNK_BYTES // (151936 * 4) == 1766
    assert get_max_chunk_logits(32000) == 8388
    assert get_max_chunk_logits(128256) == 2092
    assert get_max_chunk_logits(MAX_CHUNK_BYTES) == 1     # 至少一块，哪怕只有一行


def _cu(rows_per_req: list[int]) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(np.array(rows_per_req, dtype=np.int64))))


@pytest.mark.parametrize("chunk", [1, 4, 5, 6, 13])
def test_chunks_never_split_a_request_and_never_exceed_the_budget(chunk: int) -> None:
    """不切开请求、不超上限、无缝无重叠、一定前进（C-offset 的前提）。"""
    rng = np.random.default_rng(75)
    rows_per_req = [int(x) for x in rng.integers(1, 6, size=30)]
    cu = _cu(rows_per_req)
    chunks = list(_iter_request_chunks(cu, chunk))
    covered: list[int] = []
    for start, end in chunks:
        assert 0 <= start < end <= len(rows_per_req)
        assert int(cu[end] - cu[start]) <= chunk or end == start + 1   # 单请求超限时独占一块
        covered += list(range(start, end))
    assert covered == list(range(len(rows_per_req)))


def test_exact_fit_and_single_request_boundaries() -> None:
    """边界：正好用满上限时不多切一块；单条请求无论多大都自成一块；空批不产块。"""
    cu = _cu([2, 2, 2])                     # 每请求 2 行
    assert list(_iter_request_chunks(cu, 4)) == [(0, 2), (2, 3)]   # 0..2 正好 4 行
    assert list(_iter_request_chunks(cu, 6)) == [(0, 3)]
    assert list(_iter_request_chunks(_cu([100]), 8)) == [(0, 1)]   # 超大请求独占
    assert list(_iter_request_chunks(_cu([]), 8)) == []            # 空批


# ---------------------------------------------------------------------------
# 二、`_verify_in_chunks`：单块 vs 多块逐值一致
# ---------------------------------------------------------------------------


class _IdentitySampler:
    """只提供 `_verify` 需要的属性的替身。

    真实 V2 `Sampler.apply_sampling_params()` 在这里做的是"升 fp32 + 施加惩罚/温度/min_p"，
    与**切块算术**无关（那部分由 73 关的逐值差分覆盖）。本用例给进去的 logits 已经是"处理过"的
    那一份，所以让 `apply_sampling_params` 原样返回——这样单块与多块两条路吃到的 target logits
    逐位相同，比较的是纯粹的切片/偏移/拼接。
    """

    logprobs_mode = "raw_logprobs"
    use_fp64_gumbel = False

    def __init__(self, temperature: torch.Tensor, seeds: torch.Tensor, max_num_logprobs: int):
        self.sampling_states = SimpleNamespace(
            temperature=SimpleNamespace(gpu=temperature),
            seeds=SimpleNamespace(gpu=seeds),
            max_num_logprobs=lambda idx_mapping_np: max_num_logprobs,
        )

    def apply_sampling_params(self, logits, *_args, **_kwargs):
        return logits


def _fake_batch(drafts_per_req, slots, k, *, stale_np_extra: int = 0,
                max_num_reqs: int = MAX_NUM_REQS):
    """造一份 `_verify_in_chunks` 需要的最小批次（字段名与真 `InputBatch` 一致）。

    `stale_np_extra > 0` 时**只**把 `cu_num_logits_np`（CPU 那份，只用来挑块边界）算多
    `stale_np_extra` 行，GPU 上的 `cu_num_logits` 仍与 logits 一致——这正是 adaptive
    verification 压紧 logits 之后的状态：GPU 侧偏移是压紧后的，CPU 侧那份没跟着更新，
    于是它**不能再**用来挑块边界（075 §3.6）。
    """
    cu_np, local_pos_np, starts = layout_of(drafts_per_req)
    rows_per_req = [n + 1 for n in drafts_per_req]
    real_rows = int(cu_np[-1])
    # 自洽性：内核按 `num_speculative_steps` 判"这一行是不是 bonus 行"，草稿数超过它就是
    # 不自洽的输入（那些行的 target stats 不会被算，后面读到的就是未初始化显存 → 跨会话飘）
    assert k >= max(drafts_per_req), (k, drafts_per_req)
    stale_cu = cu_np + np.arange(len(cu_np), dtype=np.int64) * stale_np_extra
    # 草稿 token：`draft_sampled[start + i + 1]` 才是第 i 步要验证的那枚（第 0 行内核不读）
    draft_sampled = torch.zeros(real_rows, dtype=INT32, device=DEVICE)
    for r, start in enumerate(starts):
        for i in range(drafts_per_req[r]):
            draft_sampled[start + i + 1] = (r + i) % VOCAB
    # `expanded_idx_mapping` 是"按**每请求行数**展开"（不是按同一个 k+1），混合 K_i 时必须用
    # `rows_per_req` —— 用错会让某几行取到别的 slot 的温度/种子（静默错，且单块/多块不一致）
    expanded_idx_mapping = torch.tensor(
        slots, dtype=torch.int64, device=DEVICE).repeat_interleave(
        torch.tensor(rows_per_req, dtype=torch.int64, device=DEVICE))
    assert expanded_idx_mapping.numel() == real_rows == local_pos_np.size
    batch = SimpleNamespace(
        num_reqs=len(drafts_per_req),
        cu_num_logits_np=stale_cu,
        cu_num_logits=torch.from_numpy(cu_np).to(DEVICE),
        idx_mapping=torch.tensor(slots, dtype=torch.int64, device=DEVICE),
        idx_mapping_np=np.array(slots, dtype=np.int64),
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=torch.from_numpy(local_pos_np).to(DEVICE),
    )
    pos = torch.arange(200, 200 + real_rows, dtype=torch.int64, device=DEVICE)
    return batch, draft_sampled, pos, real_rows


def _chunk_case(*, drafts_per_req, slots, k=K, seed=75, with_draft_logits=True,
                stale_np_extra=0):
    """一份自洽的输入：target logits、草稿分布（按 slot）、温度/种子。"""
    gen = torch.Generator().manual_seed(seed)
    batch, draft_sampled, pos, num_logits = _fake_batch(
        drafts_per_req, slots, k, stale_np_extra=stale_np_extra)
    target_logits = (torch.randn(num_logits, VOCAB, generator=gen) * 1.5).to(DEVICE)
    draft_logits = None
    if with_draft_logits:
        draft_logits = (torch.randn(MAX_NUM_REQS, k, VOCAB, generator=gen) * 2.0).to(DEVICE)
        # 每个 slot 的草稿分布峰在不同的 token 上：误按块切 draft_logits 一定会读出别的分布
        for slot in slots:
            draft_logits[slot, :, slot % VOCAB] += 6.0
    temperature = torch.zeros(MAX_NUM_REQS, dtype=torch.float32, device=DEVICE)
    seeds = torch.zeros(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE)
    for i, slot in enumerate(slots):
        temperature[slot] = 1.0 + 0.5 * i
        seeds[slot] = 1000 + 7 * i
    return batch, target_logits, draft_logits, draft_sampled, pos, temperature, seeds


def _verify_chunks(*, chunk_limit, max_num_logprobs=-1, drafts_per_req=[K, K, K],
                   slots=[5, 0, 3], stale_np_extra=0, with_draft_logits=True, k=K):
    batch, target_logits, draft_logits, draft_sampled, pos, temperature, seeds = _chunk_case(
        drafts_per_req=drafts_per_req, slots=slots, k=k, with_draft_logits=with_draft_logits)
    if stale_np_extra:
        batch.cu_num_logits_np = batch.cu_num_logits_np + np.arange(
            len(batch.cu_num_logits_np)) * stale_np_extra
    sampler = _IdentitySampler(temperature, seeds, max_num_logprobs)
    rs = RejectionSampler(sampler, SpeculativeConfig(method="ngram",
                                                     num_speculative_tokens=k), DEVICE)
    return rs._verify_in_chunks(target_logits, batch, draft_logits, draft_sampled, pos,
                                chunk_limit, max_num_logprobs)


def _assert_same_result(a, b, k=K):
    a_sampled, a_num, a_logprobs = a
    b_sampled, b_num, b_logprobs = b
    assert torch.equal(a_num, b_num), f"{a_num.tolist()} != {b_num.tolist()}"
    assert a_sampled.shape == b_sampled.shape
    for r in range(a_num.numel()):
        n = int(a_num[r])
        assert torch.equal(a_sampled[r, :n], b_sampled[r, :n]), (r, n)
    if a_logprobs is None or b_logprobs is None:
        assert a_logprobs is None and b_logprobs is None
        return
    assert a_logprobs.cu_num_generated_tokens == b_logprobs.cu_num_generated_tokens
    assert torch.equal(a_logprobs.logprob_token_ids, b_logprobs.logprob_token_ids)
    assert torch.equal(a_logprobs.selected_token_ranks, b_logprobs.selected_token_ranks)
    assert torch.allclose(a_logprobs.logprobs, b_logprobs.logprobs, atol=1e-6)


def test_single_chunk_and_multi_chunk_agree() -> None:
    """C-offset/C-order：同一个批，单块与多块（每请求一块 / 两块）结果逐值一致。"""
    single = _verify_chunks(chunk_limit=10**9)
    # 确定性：只比**有效前缀**（`sampled` 是 `new_empty`，`num_sampled` 之后是未初始化显存，
    # 它可能带着上一个用例的残留——比它没有语义，73 关的差分夹具踩过同一个坑）
    _assert_same_result(single, _verify_chunks(chunk_limit=10**9))
    # 三种上限：每请求一块（行数 == 上限）、两块、以及"2 条请求一块"的混搭
    for limit in (K + 1, 2 * (K + 1), 3 * (K + 1) - 1):
        assert list(_iter_request_chunks(_cu([K + 1] * 3), limit))
        _assert_same_result(single, _verify_chunks(chunk_limit=limit))


def test_mixed_draft_counts_and_scrambled_slots_agree() -> None:
    """混合 K_i（含 0 草稿请求）+ 乱序 slot：多块结果与单块一致（q 仍按全局 slot 取）。"""
    drafts_per_req = [3, 0, 2, 1]
    slots = [6, 2, 5, 0]
    single = _verify_chunks(chunk_limit=10**9, drafts_per_req=drafts_per_req, slots=slots)
    for limit in (2, 3, 4, 5, 7):
        _assert_same_result(single, _verify_chunks(
            chunk_limit=limit, drafts_per_req=drafts_per_req, slots=slots), k=3)


def test_request_larger_than_the_chunk_limit_gets_its_own_chunk() -> None:
    """单条请求自己就超上限：它单独成块（绝不切开请求），结果仍与单块一致。"""
    drafts_per_req = [2, 5, 1]          # 中间那条 6 行（> 上限 4）→ 它必须单独成块
    slots = [1, 7, 4]
    k = 5                               # 必须 ≥ 最大草稿数，否则"超出 K 的行"是 bonus 行
    single = _verify_chunks(chunk_limit=10**9, drafts_per_req=drafts_per_req, slots=slots, k=k)
    _assert_same_result(single, _verify_chunks(
        chunk_limit=4, drafts_per_req=drafts_per_req, slots=slots, k=k), k=k)


def test_draft_logits_are_indexed_globally_not_per_chunk() -> None:
    """q 按**全局 slot** 取：若按块切 `draft_logits`，本用例的输出必然不同（反证）。"""
    drafts_per_req = [K, K]
    slots = [5, 6]                      # 第一块（请求 0）在全局表里的行是 5，不是 0
    single = _verify_chunks(chunk_limit=10**9, drafts_per_req=drafts_per_req, slots=slots)
    chunked = _verify_chunks(chunk_limit=K + 1, drafts_per_req=drafts_per_req, slots=slots)
    _assert_same_result(single, chunked)

    # 反证：把前两条 slot 的草稿分布互换（模拟"按块切"读到的行），输出必须变化
    batch, target_logits, draft_logits, draft_sampled, pos, temperature, seeds = _chunk_case(
        drafts_per_req=drafts_per_req, slots=slots)
    swapped = draft_logits.clone()
    swapped[5], swapped[6] = draft_logits[6].clone(), draft_logits[5].clone()
    sampler = _IdentitySampler(temperature, seeds, -1)
    rs = RejectionSampler(sampler, SpeculativeConfig(method="ngram",
                                                     num_speculative_tokens=K), DEVICE)
    counterfactual = rs._verify_in_chunks(target_logits, batch, swapped, draft_sampled, pos,
                                         K + 1, -1)
    assert not torch.equal(counterfactual[1], chunked[1]), (
        "换了 q 却没变 → 本用例没有判别力（draft_logits 根本没被读）")


def test_logprobs_survive_chunking_with_same_structure() -> None:
    """C-order：要 logprobs 时，多块拼接后的结构/顺序与单块一致（`LogprobsTensors.cat` 路径）。"""
    single = _verify_chunks(chunk_limit=10**9, max_num_logprobs=3)
    assert single[2] is not None
    for limit in (K + 1, 2 * (K + 1)):
        _assert_same_result(single, _verify_chunks(chunk_limit=limit, max_num_logprobs=3))
    # 单块与多块的返回结构一致，且 `cu_num_generated_tokens` 是**全局**请求偏移（不是块局部）
    multi = _verify_chunks(chunk_limit=K + 1, max_num_logprobs=3)
    assert single[2].cu_num_generated_tokens == [0, K + 1, 2 * (K + 1), 3 * (K + 1)]
    assert multi[2].cu_num_generated_tokens == single[2].cu_num_generated_tokens
    # 行数 = 每请求交付的 token 数之和（4 行/请求），列数 = max_num_logprobs + 1（第 0 列是采到的那个）
    assert single[2].logprob_token_ids.shape == (3 * (K + 1), 4)
    assert multi[2].logprob_token_ids.shape == (3 * (K + 1), 4)


def test_stale_cpu_offsets_do_not_pick_chunk_boundaries() -> None:
    """C-stale：`cu_num_logits_np` 陈旧（CPU 前缀和比 GPU 那份大）且行数不超上限 → 必须走**单块**。

    这条保护是 79 关（adaptive verification 压紧 logits）的前置：压紧后 GPU 侧偏移是压紧后的，
    CPU 那份前缀和还是压紧前的布局，拿它去 `searchsorted` 会切出错位的块。所以第一个分支的判据是
    `logits.shape[0] <= max_chunk_logits`，而不是"前缀和的最后一项"。
    """
    # 0) 前提：陈旧前缀和切出来的边界**确实**与新鲜的不一样（否则这条保护没有意义）。
    #    上限取 8：新鲜 [0,4,8,12] 切出 (0,2)（8 行）；陈旧 [0,6,12,18] 只能切出 (0,1)——
    #    按它算"这一块有 6 行"，可那 6 行里有 2 行根本不存在（这就是错位）。
    fresh_cu = _cu([K + 1, K + 1, K + 1])
    stale_cu = fresh_cu + np.arange(4, dtype=np.int64) * 2
    fresh_chunks = list(_iter_request_chunks(fresh_cu, 8))
    stale_chunks = list(_iter_request_chunks(stale_cu, 8))
    assert fresh_chunks != stale_chunks, (fresh_chunks, stale_chunks)
    assert fresh_chunks == [(0, 2), (2, 3)] and stale_chunks[0] == (0, 1)

    calls: list[tuple[int, int]] = []
    original = mv_rs._iter_request_chunks

    def spy(cu_num_logits, max_chunk_logits):
        for chunk in original(cu_num_logits, max_chunk_logits):
            calls.append(chunk)
            yield chunk

    mv_rs._iter_request_chunks = spy
    try:
        # 1) 真实 12 行、陈旧 CPU 前缀和最后一项 18；上限 20 → 行数 ≤ 上限 → 单块，
        #    **不调用**切块器，且结果与新鲜 offsets 的单块结果逐值一致
        stale = _verify_chunks(chunk_limit=20, stale_np_extra=2)
        assert calls == [], f"陈旧 offsets 被拿去切块了：{calls}"
        _assert_same_result(_verify_chunks(chunk_limit=10**9), stale)

        # 2) 反向：新鲜 offsets 且行数超上限时，切块器**必须**被调用（证明上面的断言不是恒真）
        _verify_chunks(chunk_limit=K + 1)
        assert calls, "超过上限却没切块"
    finally:
        mv_rs._iter_request_chunks = original


def test_chunked_path_is_bit_identical_for_one_hot_drafts() -> None:
    """点质量草稿（`draft_logits=None`）也要在切块前后逐值一致（HAS_DRAFT_LOGITS=False 分支）。"""
    drafts_per_req = [K, K, 0]
    slots = [3, 4, 6]
    single = _verify_chunks(chunk_limit=10**9, drafts_per_req=drafts_per_req, slots=slots,
                            with_draft_logits=False)
    _assert_same_result(single, _verify_chunks(chunk_limit=K + 1, drafts_per_req=drafts_per_req,
                                              slots=slots, with_draft_logits=False))


# ---------------------------------------------------------------------------
# 三、引擎层：把上限压小，真引擎跑一遍，与非切块逐 token / 逐 logprob 比
# ---------------------------------------------------------------------------


def _generate(*, chunk_limit, monkeypatch, logprobs=2, max_tokens=8):
    """tiny V2 引擎跑一个固定 workload，返回 `{req_id: (tokens, logprobs)}`。

    `chunk_limit` 是给 `get_max_chunk_logits` 的替换值（None = 真实上限，即不切块）。
    """
    if chunk_limit is not None:
        monkeypatch.setattr(mv_rs, "get_max_chunk_logits", lambda vocab: chunk_limit)
    from minivllm import SamplingParams

    engine, _core, runner = make_v2_engine(spec_k=K, budget=64, max_num_seqs=2,
                                           max_model_len=64)
    try:
        prompts = {"r0": [1, 2, 3, 4, 5], "r1": [7, 8, 9]}
        for req_id, prompt in prompts.items():
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                              seed=3, eos_token_id=VOCAB + 7,
                                              logprobs=logprobs))
        outputs: dict[str, tuple[list[int], object]] = {}
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                outputs[out.request_id] = (list(out.token_ids),
                                           getattr(out, "logprobs", None))
        num_speculative_steps = runner.rejection_sampler.num_speculative_steps
        return outputs, num_speculative_steps
    finally:
        engine.shutdown()


@requires_cuda
def test_engine_chunked_generation_matches_unchunked(monkeypatch) -> None:
    """端到端：把上限压到"每条请求各自一块"（1+K=4 行 > 上限 3），交付必须逐位相同。

    同时证明切块真的发生了（块数 > 1）：否则本用例会退化成"不切块 == 不切块"。
    """
    chunks: list[tuple[int, int]] = []
    original = mv_rs._iter_request_chunks

    def spy(cu_num_logits, max_chunk_logits):
        for chunk in original(cu_num_logits, max_chunk_logits):
            chunks.append(chunk)
            yield chunk

    monkeypatch.setattr(mv_rs, "_iter_request_chunks", spy)
    reference, _sampler_steps = _generate(chunk_limit=None, monkeypatch=monkeypatch)
    assert chunks == [], "不切块时不该调用切块器"

    chunked, sampler_steps = _generate(chunk_limit=K, monkeypatch=monkeypatch)
    assert chunks, "上限压到 K 之后仍然没切块"
    assert len(chunks) > 1

    assert set(reference) == set(chunked)
    for req_id in reference:
        ref_tokens, ref_logprobs = reference[req_id]
        got_tokens, got_logprobs = chunked[req_id]
        assert got_tokens == ref_tokens, req_id
        assert ref_logprobs is not None and got_logprobs is not None
        assert len(got_logprobs) == len(ref_logprobs)
        for pos, (ref_at_pos, got_at_pos) in enumerate(zip(ref_logprobs, got_logprobs)):
            assert set(ref_at_pos) == set(got_at_pos), (req_id, pos)
            for token_id, ref_entry in ref_at_pos.items():
                got_entry = got_at_pos[token_id]
                assert got_entry.rank == ref_entry.rank, (req_id, pos, token_id)
                assert got_entry.logprob == pytest.approx(ref_entry.logprob, abs=1e-6)
    # 采样器确实按小上限工作（证明上面用的是同一条 `_verify_in_chunks`）
    assert sampler_steps == K
