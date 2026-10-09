"""73 关：V2 logprobs / 采样掩码的**逐值差分**——本项目移植 vs 上游 vLLM 0.28.0。

被测对象（两个文件都是"逐行移植"，只改 import 与点名的等价工具）：

- 本项目：``minivllm/worker/gpu/sample/logprob.py``
  ↔ 上游 ``vllm/v1/worker/gpu/sample/logprob.py``
- 本项目：``minivllm/worker/gpu/sample/output.py``
  ↔ 上游 ``vllm/v1/worker/gpu/sample/output.py``（外加从上游 ``v1/outputs.py`` 搬来的
  ``SamplingMaskLists``）

**为什么可以要求逐值相等**：两边拿到的是同一批张量、同一份 ``cu_num_logits``／
``expanded_idx_mapping``，内核里没有原子操作、也没有全局随机状态，所以只要移植没改语义，
``logprob_token_ids``／``selected_token_ranks`` 就必须**逐位相同**。``logprobs`` 是 fp32
的 ``logit - max - log(Σexp)``，块内 ``tl.sum`` 树形归约、块间顺序累加，因此按 fp32 归约
顺序差异给 ``atol=1e-6``（理由写在用例里），**不放宽到 1e-3 这种量级**：差分不过就是移植
错了，要修代码而不是放宽断言。

环境变量说明：``LogprobTokenIdsState`` 的 ``num_token_ids`` 是 ``UvaBackedTensor``（UVA 表），
而 ``is_pin_memory_available()`` 在本机（WSL2）默认返回 False，判定又被 ``functools.cache``
缓存——所以必须在 **import minivllm 之前**设 ``VLLM_WSL2_ENABLE_PIN_MEMORY=1``（与 AGENTS §3
里跑 V2 通路的要求一致，不是本测试的额外要求）。没设的话 ``UvaBuffer`` 会直接报
"UVA is not available"。

设备：``"cuda" if torch.cuda.is_available() else "cpu"``。用 Triton 内核的用例（topk/rank/
打包掩码）在 CUDA 上真跑，没有 CUDA 时按用例跳过；``LogprobTokenIdsState`` 的 CPU 记账逻辑
在两种设备上都跑。
"""

from __future__ import annotations

import os
from types import SimpleNamespace

# 必须早于 `import minivllm`（`is_uva_available()` 被 functools.cache 缓存，见模块 docstring）。
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import numpy as np
import pytest
import torch

import minivllm.worker.gpu.sample.logprob as mv_logprob
import minivllm.worker.gpu.sample.output as mv_output
import vllm.v1.worker.gpu.sample.logprob as up_logprob
import vllm.v1.worker.gpu.sample.output as up_output
from minivllm.sampling_params import SamplingParams as MvSamplingParams

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_cuda = pytest.mark.skipif(
    DEVICE != "cuda", reason="Triton 内核（topk/rank/掩码打包）只在 CUDA 上跑"
)

# 小配置：3 条请求、词表 64、每条要 4 个 logprobs（形状与上游 docstring 一致）。
NUM_REQS = 3
VOCAB = 64
NUM_LOGPROBS = 4
# cu_num_logits 是"每请求的 logits 行区间"（前闭后开），带前导 0：3 条请求各 1 行。
CU_NUM_LOGITS = [0, 1, 2, 3]


def _logits_and_sampled(seed: int = 73) -> tuple[torch.Tensor, torch.Tensor]:
    """固定 seed 的 logits（[num_reqs, vocab]）与采到的 token（[num_reqs]）。

    logits 用连续 fp32（内核按 `logits.stride(0)` 寻址）；采到的 token 故意与 argmax 不同
    （第 0 列是"采到的那个"，其余列才是 top-k，两者错位才测得出列语义）。
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(NUM_REQS, VOCAB, generator=g, dtype=torch.float32).to(DEVICE)
    sampled = torch.tensor([3, 41, 17], dtype=torch.int64, device=DEVICE)
    return logits, sampled


# ---------------------------------------------------------------------------
# 1) compute_topk_scores：与上游逐值差分（本文件的核心）
# ---------------------------------------------------------------------------


@requires_cuda
def test_compute_topk_scores_matches_upstream():
    logits, sampled = _logits_and_sampled()

    mv = mv_logprob.compute_topk_scores(
        logits, NUM_LOGPROBS, sampled, cu_num_logits=CU_NUM_LOGITS
    )
    up = up_logprob.compute_topk_scores(
        logits, NUM_LOGPROBS, sampled, cu_num_logits=CU_NUM_LOGITS
    )

    # 布局：第 0 列 = 采到的 token，其后 NUM_LOGPROBS 列 = top-k。
    assert mv.logprob_token_ids.shape == (NUM_REQS, NUM_LOGPROBS + 1)
    assert mv.logprobs.shape == (NUM_REQS, NUM_LOGPROBS + 1)
    assert mv.selected_token_ranks.shape == (NUM_REQS,)
    assert mv.cu_num_generated_tokens == CU_NUM_LOGITS

    # token id 与 rank 是整数量，必须逐位相等。
    assert torch.equal(mv.logprob_token_ids, up.logprob_token_ids)
    assert torch.equal(mv.selected_token_ranks, up.selected_token_ranks)
    assert mv.logprob_token_ids.dtype == up.logprob_token_ids.dtype
    assert mv.selected_token_ranks.dtype == up.selected_token_ranks.dtype

    # logprobs 用 atol=1e-6 的理由：分数 = logit - max - log(Σexp)，Σ 在 BLOCK_SIZE 内由
    # `tl.sum` 做树形归约、块间顺序累加；换编译特化/归约顺序时误差量级是 O(√n·eps_fp32)，
    # n=vocab=64、eps=2^-24≈6e-8 → 最坏 ≈ 5e-7，取 1e-6 这个最小整数档即可覆盖。
    # 本次实测 max|Δ|=0（同一份内核、同一份输入，无原子操作 → 逐位相同），
    # 但断言仍按上面的归约误差口径写，不把"恰好逐位相同"当成移植要求。
    max_abs = (mv.logprobs.double() - up.logprobs.double()).abs().max().item()
    assert torch.allclose(mv.logprobs, up.logprobs, atol=1e-6, rtol=0), (
        f"logprobs 与上游不一致：max|Δ|={max_abs}"
    )

    # 防"两边同错"：再用公式独立验一遍（fp64 参考），顺带钉住列语义与 rank 口径。
    ref = torch.log_softmax(logits.double(), dim=-1).gather(
        1, mv.logprob_token_ids.long()
    )
    assert torch.allclose(mv.logprobs.double(), ref, atol=1e-5, rtol=0)
    assert torch.equal(mv.logprob_token_ids[:, 0], sampled)
    # `_ranks_kernel` 数的是 `logits >= x`（含自己）→ rank 是 1-based 名次。
    ref_ranks = (logits >= logits.gather(1, sampled.unsqueeze(1))).sum(dim=-1)
    assert torch.equal(mv.selected_token_ranks, ref_ranks)


@requires_cuda
def test_compute_topk_scores_zero_logprobs_matches_upstream():
    """`num_logprobs=0`：只交付"采到的那个"（形状 [B,1]），top-k 那支整个不执行。"""
    logits, sampled = _logits_and_sampled(seed=74)

    mv = mv_logprob.compute_topk_scores(logits, 0, sampled, cu_num_logits=CU_NUM_LOGITS)
    up = up_logprob.compute_topk_scores(logits, 0, sampled, cu_num_logits=CU_NUM_LOGITS)

    assert mv.logprob_token_ids.shape == (NUM_REQS, 1)
    assert torch.equal(mv.logprob_token_ids, up.logprob_token_ids)
    assert torch.equal(mv.logprob_token_ids[:, 0], sampled)
    assert torch.equal(mv.selected_token_ranks, up.selected_token_ranks)
    assert torch.allclose(mv.logprobs, up.logprobs, atol=1e-6, rtol=0)


@requires_cuda
def test_compute_topk_scores_logits_mode_matches_upstream():
    """`logits_mode=True`（raw logits 模式）：分数直接取 logits，不走 log-softmax 内核。"""
    logits, sampled = _logits_and_sampled(seed=75)

    mv = mv_logprob.compute_topk_scores(
        logits, NUM_LOGPROBS, sampled, cu_num_logits=CU_NUM_LOGITS, logits_mode=True
    )
    up = up_logprob.compute_topk_scores(
        logits, NUM_LOGPROBS, sampled, cu_num_logits=CU_NUM_LOGITS, logits_mode=True
    )

    assert torch.equal(mv.logprob_token_ids, up.logprob_token_ids)
    assert torch.equal(mv.selected_token_ranks, up.selected_token_ranks)
    assert torch.allclose(mv.logprobs, up.logprobs, atol=1e-6, rtol=0)
    # raw 模式下分数就是被选中列的 logits（逐位相同，不需要容差）。
    assert torch.equal(
        mv.logprobs, logits.gather(1, mv.logprob_token_ids.long()).to(torch.float32)
    )


@requires_cuda
def test_compute_topk_scores_custom_token_ids_matches_upstream():
    """慢路径（`max_per_req_token_ids > 0`）：走 `_fill_logprob_token_ids_kernel`。

    本项目 `SamplingParams` 在请求期拒绝 `logprob_token_ids`（068 §3.5），所以这里直接构造
    两边的 `LogprobTokenIdsState`（上游 Runner 就是这么喂慢路径的），逐值差分整个分支：
    有自定义 token 的请求用自定义列、没有的请求回落 top-k，无效列被置 -inf。
    """
    logits, sampled = _logits_and_sampled(seed=76)
    device = torch.device(DEVICE)

    mv_state = mv_logprob.LogprobTokenIdsState(NUM_REQS, device)
    up_state = up_logprob.LogprobTokenIdsState(NUM_REQS, device)
    per_req = [[5, 9, 60], [], [31]]
    for req_idx, ids in enumerate(per_req):
        stub = SimpleNamespace(logprob_token_ids=ids)  # 只被读 `logprob_token_ids`
        mv_state.add_request(req_idx, stub)
        up_state.add_request(req_idx, stub)
    mv_state.apply_staged_writes()
    up_state.apply_staged_writes()

    max_per_req = max(len(ids) for ids in per_req)
    expanded_idx_mapping = torch.arange(NUM_REQS, dtype=torch.int32, device=device)
    kwargs = dict(
        cu_num_logits=CU_NUM_LOGITS,
        logprob_token_ids_state=mv_state,
        expanded_idx_mapping=expanded_idx_mapping,
        max_per_req_token_ids=max_per_req,
    )
    mv = mv_logprob.compute_topk_scores(logits, NUM_LOGPROBS, sampled, **kwargs)
    up_kwargs = dict(kwargs, logprob_token_ids_state=up_state)
    up = up_logprob.compute_topk_scores(logits, NUM_LOGPROBS, sampled, **up_kwargs)

    # num_cols = max(num_logprobs, max_per_req_token_ids) = 4 → 1 + 4 列。
    assert mv.logprob_token_ids.shape == (NUM_REQS, 1 + NUM_LOGPROBS)
    assert torch.equal(mv.logprob_token_ids, up.logprob_token_ids)
    assert torch.equal(mv.selected_token_ranks, up.selected_token_ranks)
    assert torch.allclose(mv.logprobs, up.logprobs, atol=1e-6, rtol=0)
    assert torch.equal(torch.isinf(mv.logprobs), torch.isinf(up.logprobs))

    # 逐请求核对列语义（防"两边同错"）：
    ids = mv.logprob_token_ids.cpu()
    scores = mv.logprobs.cpu()
    topk = torch.topk(logits.cpu(), NUM_LOGPROBS, dim=-1).indices
    for req_idx, custom in enumerate(per_req):
        assert ids[req_idx, 0].item() == sampled[req_idx].item()
        if custom:
            # 自定义 token 覆盖 top-k 列，剩下的列无效（id=0 且分数被 masked_fill 成 -inf）。
            assert ids[req_idx, 1 : 1 + len(custom)].tolist() == custom
            assert torch.isinf(scores[req_idx, 1 + len(custom) :]).all()
            assert (scores[req_idx, 1 + len(custom) :] < 0).all()
        else:
            assert ids[req_idx, 1:].tolist() == topk[req_idx].tolist()
            assert torch.isfinite(scores[req_idx, 1:]).all()


# ---------------------------------------------------------------------------
# 2) LogprobTokenIdsState：逐请求长度表 / token 表
# ---------------------------------------------------------------------------


def test_logprob_token_ids_state_tables_and_leading_rejection():
    """CPU 逻辑：`add_request` 后内部表与 `logprob_token_ids` 一致。

    ⚠️ 本项目 `SamplingParams.logprob_token_ids` 是**请求期拒绝**的字段（068 §3.5 三态矩阵里
    属"本项目尚未接入"）：`SamplingParams(logprob_token_ids=[...])` 直接 `NotImplementedError`
    （下面用 `pytest.raises` 钉住）。所以状态类的 CPU 逻辑用**只带该属性的最小替身**测
    （`add_request` 只读 `sampling_params.logprob_token_ids`，上游同款），
    真正的上游 `SamplingParams` 交互在下一个用例（CUDA 差分）里验证。
    """
    with pytest.raises(NotImplementedError, match="logprob_token_ids"):
        MvSamplingParams(logprob_token_ids=[1, 2, 3])

    state = mv_logprob.LogprobTokenIdsState(max_num_reqs=4, device=torch.device(DEVICE))
    state.add_request(0, SimpleNamespace(logprob_token_ids=[5, 9, 11]))
    state.add_request(1, SimpleNamespace(logprob_token_ids=None))
    state.add_request(2, SimpleNamespace(logprob_token_ids=[]))
    state.add_request(3, SimpleNamespace(logprob_token_ids=[64]))

    # 长度表（UVA 的 CPU 侧就是真相）：与每请求给的长度逐个一致。
    np.testing.assert_array_equal(
        state.num_token_ids.np[:4], np.array([3, 0, 0, 1], dtype=np.int32)
    )
    idx_mapping_np = np.array([0, 1, 2, 3], dtype=np.int64)
    assert state.max_num_token_ids(idx_mapping_np) == 3
    # 只看这个 batch 里出现的行号（上游 Runner 传的就是 idx_mapping_np 切片）。
    assert state.max_num_token_ids(np.array([1, 2], dtype=np.int64)) == 0
    assert state.max_num_token_ids(np.array([], dtype=np.int64)) == 0

    # 落盘后 token 表逐行等于给的值，未写的位置保持 0（表宽 = MAX_LOGPROB_TOKEN_IDS）。
    state.apply_staged_writes()
    rows = state.token_ids.gpu.cpu().numpy()
    assert rows.shape == (4, mv_logprob.MAX_LOGPROB_TOKEN_IDS)
    np.testing.assert_array_equal(rows[0, :3], np.array([5, 9, 11], dtype=np.int32))
    np.testing.assert_array_equal(rows[3, :1], np.array([64], dtype=np.int32))
    assert (rows[1] == 0).all() and (rows[2] == 0).all()
    assert (rows[0, 3:] == 0).all() and (rows[3, 1:] == 0).all()


def test_logprob_token_ids_state_max_length_boundary():
    """`MAX_LOGPROB_TOKEN_IDS` 的口径与上游一致（=128）：等于上限收下，超过上限报错。"""
    assert mv_logprob.MAX_LOGPROB_TOKEN_IDS == 128
    state = mv_logprob.LogprobTokenIdsState(max_num_reqs=2, device=torch.device(DEVICE))

    exact = list(range(mv_logprob.MAX_LOGPROB_TOKEN_IDS))
    state.add_request(0, SimpleNamespace(logprob_token_ids=exact))
    assert state.num_token_ids.np[0] == mv_logprob.MAX_LOGPROB_TOKEN_IDS
    state.apply_staged_writes()
    np.testing.assert_array_equal(
        state.token_ids.gpu.cpu().numpy()[0], np.array(exact, dtype=np.int32)
    )

    with pytest.raises(ValueError, match="Too many logprob_token_ids"):
        state.add_request(1, SimpleNamespace(logprob_token_ids=exact + [1]))
    # 报错的这次没有写长度表（检查在写表之前，上游同款顺序）。
    assert state.num_token_ids.np[1] == 0


@requires_cuda
def test_logprob_token_ids_state_matches_upstream():
    """与上游状态类逐值差分（用上游真 `SamplingParams` 喂两边）。"""
    from vllm.sampling_params import SamplingParams as UpSamplingParams

    device = torch.device(DEVICE)
    mv_state = mv_logprob.LogprobTokenIdsState(4, device)
    up_state = up_logprob.LogprobTokenIdsState(4, device)

    # 上游接受 `logprob_token_ids`；本项目在请求期拒绝——差分时两边喂同一份"上游参数"，
    # 状态类只读 `logprob_token_ids` 这一个属性，所以语义可比。
    params = {
        0: UpSamplingParams(logprob_token_ids=[4, 7, 63]),
        1: UpSamplingParams(logprob_token_ids=None),
        2: UpSamplingParams(logprob_token_ids=[1]),
        3: UpSamplingParams(logprob_token_ids=[]),
    }
    for req_idx, sp in params.items():
        mv_state.add_request(req_idx, sp)
        up_state.add_request(req_idx, sp)

    np.testing.assert_array_equal(mv_state.num_token_ids.np, up_state.num_token_ids.np)
    for idx_mapping_np in (np.array([0, 1, 2, 3]), np.array([3, 2]), np.array([1])):
        assert mv_state.max_num_token_ids(idx_mapping_np) == up_state.max_num_token_ids(
            idx_mapping_np
        )

    mv_state.apply_staged_writes()
    up_state.apply_staged_writes()
    np.testing.assert_array_equal(
        mv_state.token_ids.gpu.cpu().numpy(), up_state.token_ids.gpu.cpu().numpy()
    )


# ---------------------------------------------------------------------------
# 3) SamplingMaskTensors：finite-logit 支持集 + CSR（tolists）
# ---------------------------------------------------------------------------

MASK_VOCAB = 24
# 逐请求构造成员：-inf / +inf 都要被排除（上游 `keep = (logits > -inf) & (logits < inf)`）。
# 行 0 排除 {0, 1, 7, 8, 23} → 19 个 finite；行 1 全 finite 但**没采样**（inactive）；
# 行 2 排除 {5(-inf), 6(+inf)} → 22 个 finite。
MASK_EXCLUDED = {0: {0, 1, 7, 8, 23}, 1: set(), 2: {5, 6}}
MASK_NUM_SAMPLED = [1, 0, 1]


def _mask_logits() -> torch.Tensor:
    logits = torch.zeros(len(MASK_NUM_SAMPLED), MASK_VOCAB, dtype=torch.float32, device=DEVICE)
    for req_idx, excluded in MASK_EXCLUDED.items():
        for token_id in excluded:
            logits[req_idx, token_id] = float("inf") if token_id == 6 else float("-inf")
    return logits


def _hand_support(req_idx: int) -> list[int]:
    return [t for t in range(MASK_VOCAB) if t not in MASK_EXCLUDED[req_idx]]


def _hand_packed_bytes(req_idx: int) -> list[int]:
    """手工按"每 8 个 token 一个字节、低位在前（bit i = token i）"打包 finite 支持集。

    内核里的位是 `(finite & is_active)`：**没采样的行整行不置位**（`counts` 也是 0），
    所以这里同样乘上 `num_sampled_tokens > 0`。
    """
    active = MASK_NUM_SAMPLED[req_idx] > 0
    packed = []
    for byte_start in range(0, MASK_VOCAB, 8):
        value = 0
        for bit in range(8):
            token_id = byte_start + bit
            if active and token_id < MASK_VOCAB and token_id in _hand_support(req_idx):
                value |= 1 << bit
        packed.append(value)
    return packed


@requires_cuda
def test_sampling_mask_tensors_from_logits_counts_and_bits():
    logits = _mask_logits()
    num_sampled = torch.tensor(MASK_NUM_SAMPLED, dtype=torch.int32, device=DEVICE)

    mv = mv_output.SamplingMaskTensors.from_logits(logits, num_sampled)
    up = up_output.SamplingMaskTensors.from_logits(logits, num_sampled)

    assert mv.vocab_size == MASK_VOCAB == up.vocab_size
    assert mv.packed_mask.shape == (len(MASK_NUM_SAMPLED), (MASK_VOCAB + 7) // 8)
    assert mv.counts.dtype == torch.int32

    # 逐值差分（`_pack_sampling_mask_kernel` 是同一个内核体）。
    assert torch.equal(mv.packed_mask, up.packed_mask)
    assert torch.equal(mv.counts, up.counts)

    # 手工数出来的 finite 个数：inactive 行（没采样）恒为 0。
    expected_counts = [
        len(_hand_support(i)) if MASK_NUM_SAMPLED[i] else 0
        for i in range(len(MASK_NUM_SAMPLED))
    ]
    assert expected_counts == [19, 0, 22]  # 手算：24-5 / (inactive) / 24-2
    assert mv.counts.cpu().tolist() == expected_counts
    # 打包字节也手工对一遍（钉住"低位在前、跨字节续接"的布局）。
    assert mv.packed_mask.cpu().tolist() == [_hand_packed_bytes(i) for i in range(3)]
    assert _hand_packed_bytes(0) == [0x7C, 0xFE, 0x7F]
    assert _hand_packed_bytes(1) == [0x00, 0x00, 0x00]  # inactive：整行不置位
    assert _hand_packed_bytes(2) == [0x9F, 0xFF, 0xFF]  # 排除的是 {5,6}，所以 bit 7 仍然置位
    # 行 1 是 inactive：字节全 0、计数 0（不是"全 finite"）。
    assert mv.packed_mask.cpu()[1].tolist() == [0, 0, 0]


@requires_cuda
def test_sampling_mask_tensors_tolists_csr_matches_upstream_and_hand():
    logits = _mask_logits()
    num_sampled = torch.tensor(MASK_NUM_SAMPLED, dtype=torch.int32, device=DEVICE)
    num_sampled_np = np.array(MASK_NUM_SAMPLED, dtype=np.int32)

    mv = mv_output.SamplingMaskTensors.from_logits(logits, num_sampled)
    up = up_output.SamplingMaskTensors.from_logits(logits, num_sampled)

    mv_lists = mv.tolists(num_sampled_np)
    up_lists = up.tolists(num_sampled_np)

    # 逐值差分：CSR 的两个数组 + 每请求位置偏移。
    np.testing.assert_array_equal(mv_lists.token_ids, up_lists.token_ids)
    np.testing.assert_array_equal(mv_lists.offsets, up_lists.offsets)
    assert mv_lists.cu_num_generated_tokens == up_lists.cu_num_generated_tokens

    # 手工核对 CSR 结构：只有 sampled 行（flatnonzero 的 [0, 2]）进 CSR。
    hand_token_ids = _hand_support(0) + _hand_support(2)
    assert mv_lists.token_ids.tolist() == hand_token_ids
    assert mv_lists.token_ids.dtype == np.int32
    assert mv_lists.offsets.tolist() == [0, 19, 41]
    assert mv_lists.cu_num_generated_tokens == [0, 1, 1, 2]  # cumsum([0,1,0,1])
    # 结构不变式：offsets 单调不减、首项 0、末项 == 支持集总数、段数 == 进 CSR 的行数。
    assert (np.diff(mv_lists.offsets) >= 0).all()
    assert mv_lists.offsets[0] == 0
    assert mv_lists.offsets[-1] == len(mv_lists.token_ids)
    assert len(mv_lists.offsets) == int(np.count_nonzero(num_sampled_np)) + 1
    # token_ids 全是 finite 位置、每段严格递增（np.nonzero 的行内顺序）。
    for start, end in zip(mv_lists.offsets[:-1], mv_lists.offsets[1:]):
        segment = mv_lists.token_ids[start:end]
        assert len(segment) == 0 or (np.diff(segment) > 0).all()

    # `to_cpu_nonblocking()` 只搬设备，不改数据（异步 D2H 句柄的语义）。
    cpu_copy = mv.to_cpu_nonblocking()
    assert cpu_copy.packed_mask.device.type == "cpu"
    assert torch.equal(cpu_copy.packed_mask, mv.packed_mask.cpu())
    assert torch.equal(cpu_copy.counts, mv.counts.cpu())
    assert cpu_copy.vocab_size == MASK_VOCAB
    assert cpu_copy.tolists(num_sampled_np).offsets.tolist() == mv_lists.offsets.tolist()
