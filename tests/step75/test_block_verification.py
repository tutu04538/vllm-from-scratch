"""75 关：**块验证**（block verification，Sun et al. 2024，arXiv:2403.10444）。

被测对象：`minivllm/worker/gpu/spec_decode/rejection_sampler_utils.py` 的块验证分支
（`_compute_cumulative_log_p_kernel` → `_compute_local_residual_mass_kernel` →
`_rejection_kernel` 的 `USE_BLOCK_VERIFICATION` 分支 → `_resample_kernel` 的块残差）。

三方对照（需求 075 §4 要求"Torch 数学参考、冻结 V2 kernel、本项目三方比较；统计输出分布，
不能只比接受率"）：

    冻结实现   `vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils`（site-packages 的
               vllm==0.28.0 = 需求里的源码基线）
    Torch 参考 `minivllm/testing/torch_block_verification.py`（照源码递推写的数学参考）
    本项目     `minivllm/.../rejection_sampler_utils.py`（逐行移植）

三类断言，各有分工：

1. **逐值差分**（前两组用例）：同输入、同随机源（内核的随机数就是 `tl_rand32(seed, pos)` 与
   gumbel 竞赛，无全局状态）→ `sampled`/`num_sampled` 必须逐值相等；并且**逐个 kernel** 比中间
   缓冲（累计对数比、局部残差质量、被拒位的 logsumexp），这样"哪一段移植错了"能直接定位。
2. **分布对照**（残差与接受长度）：内核的 gumbel/均匀数与参考实现不是同一套随机源，所以这里比
   **分布**（总变差 + 卡方），不比逐值（总纲 §5）。用"一次调用里塞几千个请求、每请求一个不同
   seed"的办法把统计量做足：8000 个请求一次调用约 0.17 s。
3. **分支是否真的活着**：同一批输入分别跑 block 与 standard，断言两者的输出**必须不同**，
   并给出"若把块残差写成 standard 的 `max(target-draft,0)` 就会被卡方抓住"的反证
   （实测 TV=0.19，远大于噪声）。

### 本文件记下的一个实测事实（不是断言，见 docs/step75_alignment.md §4）

"块验证的每轮期望产出永不差于标准"在论文的实验区间（草稿接近 target）成立；在**对抗性合成分布**
（草稿与 target 差得远）下本实现实测会更差：

    mix（草稿与 target 的接近程度） 0.00  0.25  0.50  0.75  0.90  1.00
    E[τ] block                     0.303 0.563 1.433 2.472 2.904 3.000
    E[τ] standard                  0.333 0.594 1.356 2.352 2.836 3.000

（K=3、V=8、随机 logits、200 组设计取平均；`tests/step75` 只断言"与源码递推一致"，
不把论文的性能主张当验收条件——需求 075 §5 要求以源码递推为实现基准。）
"""

from __future__ import annotations

import pytest
import torch

from spec75_helpers import (  # noqa: E402  （pytest rootdir 会把本目录加进 sys.path）
    DEVICE,
    INT32,
    INT64,
    build_v2_inputs,
    layout_of,
    peak_logits,
    requires_cuda,
)
from minivllm.testing.torch_block_verification import (  # noqa: E402
    BlockCase,
    total_variation,
)

pytestmark = requires_cuda

if DEVICE != "cuda":  # pragma: no cover - requires_cuda 已经整模块跳过
    pytest.skip("块验证是 Triton 内核，只在 CUDA 上可执行", allow_module_level=True)

import minivllm.worker.gpu.spec_decode.rejection_sampler_utils as mv_rs  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as up_rs  # noqa: E402

VOCAB = 8
MAX_NUM_REQS = 8


# ---------------------------------------------------------------------------
# 构造用例：手写的小 logits（可验算）
# ---------------------------------------------------------------------------

#: "残差例"：K=2、带草稿分布、草稿高估了自己的 token（p_0 < 1）→ 块残差要按 `p_tau` 缩放。
#: 手算（V=8）：
#:   行 0 target 全 0（每 token 0.125）；行 0 draft 在 t1 上 +0.6（q(t1)=0.2065）
#:     → d_0 = t1、p_0 = 0.125/0.2065 = 0.6052
#:   行 1 target = [1.0,0.5,0,-1,0,0.5,1.5,0]；行 1 draft 在 t3 上 +2.0（q(t3)=0.5135）
#:     → d_1 = t3、M_0 = Σ max(p_0·M_b - M_s, 0) = 0.1801 → h_0 = 0.1801/(0.1801+1-0.6052) = 0.3134
#:     → p_1 = 0.6052·(M_b(t3)/M_s(t3)) = 0.0313 = h_1（最后一位，阈值就是 p_i）
#:   τ 分布 = [0.6651, 0.3036, 0.0313]；τ=1 的残差两者形状明显不同：
#:     block    = max(p_0·M_b - M_s, 0) 归一化 → [0.2728, 0.0137, 0, 0, 0, 0.0137, 0.6999, 0]
#:     standard = max(M_b - M_s, 0)     归一化 → [0.2599, 0.1015, 0.0054, 0,
#:                                              0.0054, 0.1015, 0.521, 0.0054]
#:     TV = 0.1917（这就是"两者必须被区分开"的判据）
RESIDUAL_TARGET = [
    [0.0] * VOCAB,
    [1.0, 0.5, 0.0, -1.0, 0.0, 0.5, 1.5, 0.0],
    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]
RESIDUAL_DRAFT = [
    [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
]
RESIDUAL_TOKENS = [1, 3]


def residual_case() -> BlockCase:
    return BlockCase(torch.tensor(RESIDUAL_TARGET, dtype=torch.float32),
                     torch.tensor(RESIDUAL_DRAFT, dtype=torch.float32),
                     list(RESIDUAL_TOKENS), 1.0)


def one_hot_case() -> BlockCase:
    """点质量草稿（`draft_logits=None`）：残差质量走闭式 `p_i * (1 - M_b(d_{i+1}))`。

    手算（V=8、K=3）：行 0 target 峰在 t1（0.5），d_0=t1 → p_0=0.5；
    行 1 target 全 0（0.125），d_1=t2 → M_0=0.5·(1-0.125)=0.4375 → h_0=0.4375/(0.4375+0.5)=0.4667；
    行 2 target 峰在 t5（0.5）+t1（0.25），d_2=t5 → p_1=0.5·0.125=0.0625、M_1=0.0625·(1-0.5)=0.03125
    → h_1=0.03125/(0.03125+0.9375)=0.0323、h_2=p_2=0.0625·0.5=0.03125；
    行 3（bonus）target 峰在 t6 → τ=3 时从它采 bonus。τ 分布四个分支都非零。
    """
    target = torch.full((4, VOCAB), 0.0, dtype=torch.float32)
    target[0, 1] = 2.0
    target[2, 5] = 2.0
    target[2, 1] = 0.5
    target[3, 6] = 3.0
    return BlockCase(target, None, [1, 2, 5], 1.0)


def bonus_case() -> BlockCase:
    """K=1：既是"最后一位"（阈值 = p_i，没有下一行）也是"全接受就发 bonus"的边界。

    行 0 target 峰在 t3；行 1（bonus）target 峰在 t4；点质量草稿 d_0=t3 → p_0=p(t3)=0.5、
    h_0=p_0（最后候选）→ τ=0/1 各半：τ=0 走"target 去掉被拒 token"，τ=1 走 bonus 分布。
    """
    target = torch.zeros(2, VOCAB, dtype=torch.float32)
    target[0, 3] = 1.0
    target[1, 4] = 2.0
    return BlockCase(target, None, [3], 1.0)


def zero_prob_case() -> BlockCase:
    """首接受长度为 0 的确定性构造：d_0 的 target 概率是 0（logit=-inf）→ p_0 = 0 → h_0 = 0。

    内核在 `u <= h` 上用 `u > 0`（`tl_rand32(..., includes_zero=False)`）→ 必拒；
    块残差在 τ=0 时 `p_tau=1`，退化成 standard 的"target 去掉被拒草稿 token"。
    """
    target = torch.zeros(2, VOCAB, dtype=torch.float32)
    target[0, 3] = 1.0
    target[0, 5] = float("-inf")
    target[1, 6] = 1.0
    return BlockCase(target, None, [5], 1.0)


def perfect_case(k: int = 3) -> BlockCase:
    """`p == q`：草稿分布与 target 逐位相同（`M = 0`、`p_i = 1`）→ 阈值走 `denom<=0` 的兜底 1.0。

    这条钉的是数值稳定性：`h = M / (M + 1 - p_i)` 在 `M=0, p_i=1` 时是 0/0，内核必须给 1.0
    （`tl.where(denom > 0, ..., 1.0)`），否则"完美的草稿"会被全部拒掉。
    """
    gen = torch.Generator().manual_seed(75)
    target = torch.randn(k + 1, VOCAB, generator=gen) * 2.0
    draft = target[:k].clone()
    return BlockCase(target, draft, [int(draft[i].argmax()) for i in range(k)], 1.0)


# ---------------------------------------------------------------------------
# 批量驱动：一次调用塞 N 条相同设计、不同 seed 的请求（把统计量做足）
# ---------------------------------------------------------------------------


def run_trials(case: BlockCase, n_trials: int, *, module=mv_rs, use_std: bool = False,
               seed_base: int = 0, pos_start: int = 100):
    """把 `case` 复制成 `n_trials` 条请求跑一次内核，返回 `(sampled, num_sampled)`。

    每请求一个不同的 `seed`（内核的 `u = tl_rand32(seed, pos)`、gumbel 同样按 (seed,pos)），
    所以这等价于"同一分布独立抽 `n_trials` 次"。
    """
    k = case.num_drafts
    rows = k + 1
    target = case.target_logits.to(DEVICE).repeat(n_trials, 1).contiguous()
    draft_logits = None
    if case.draft_logits is not None:
        draft_logits = torch.zeros(n_trials, k, VOCAB, dtype=torch.float32, device=DEVICE)
        draft_logits[:] = case.draft_logits.to(DEVICE)
    draft_sampled = torch.zeros(n_trials * rows, dtype=INT32, device=DEVICE)
    for i, token in enumerate(case.draft_tokens):
        draft_sampled[i + 1::rows] = token
    cu = torch.arange(0, (n_trials + 1) * rows, rows, dtype=INT32, device=DEVICE)
    pos = torch.arange(0, n_trials * rows, dtype=INT64, device=DEVICE) + pos_start
    idx = torch.arange(n_trials, dtype=INT64, device=DEVICE)
    exp_idx = idx.repeat_interleave(rows)
    exp_pos = torch.arange(rows, dtype=INT32, device=DEVICE).repeat(n_trials)
    temp = torch.full((n_trials,), case.temperature, dtype=torch.float32, device=DEVICE)
    seeds = torch.arange(n_trials, dtype=INT64, device=DEVICE) + seed_base
    sampled, num_sampled = module.rejection_sample(
        target, draft_logits, draft_sampled, cu, pos, idx, exp_idx, exp_pos, temp, seeds,
        k, None, use_fp64=False, use_block_verification=not use_std)
    return sampled, num_sampled


def empirical_acceptance_distribution(num_sampled: torch.Tensor, k: int) -> torch.Tensor:
    tau = (num_sampled.to(torch.int64) - 1).cpu()
    return torch.bincount(tau, minlength=k + 1).double() / num_sampled.numel()


def conditional_token_histogram(sampled, num_sampled, tau_wanted: int) -> torch.Tensor:
    """`P(交付的最后一枚 token | τ = tau_wanted)` 的经验分布。

    交付的最后一枚写在 `sampled[r, num_sampled[r] - 1]`（拒绝内核把前 τ 位写成草稿 token，
    `_insert_resampled_kernel` 把恢复/bonus token 写在**自增之前**的 `num_sampled` 上，
    也就是下标 τ = `num_sampled - 1`）。
    """
    mask = num_sampled.to(torch.int64) == tau_wanted + 1
    count = int(mask.sum())
    assert count > 200, f"τ={tau_wanted} 的样本只有 {count} 个，统计量不足（加大 n_trials）"
    rows = torch.arange(num_sampled.numel(), device=sampled.device)[mask]
    tokens = sampled[rows, num_sampled.to(torch.int64)[mask] - 1].to(torch.int64).cpu()
    return torch.bincount(tokens, minlength=VOCAB).double() / count


def chi_square(empirical: torch.Tensor, expected: torch.Tensor) -> float:
    return float(((empirical - expected) ** 2 / expected.clamp_min(1e-12)).sum())


# ---------------------------------------------------------------------------
# ① 逐值差分：本项目 vs 冻结实现
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_draft_logits", [False, True])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_block_matches_frozen_kernel(k: int, with_draft_logits: bool) -> None:
    """同输入同随机源 → `sampled`/`num_sampled` 逐值相等（K=1/2/3、点质量与概率草稿）。"""
    gen = torch.Generator().manual_seed(750 + k)
    drafts_per_req = [k, k, k]
    slots = [5, 0, 3]
    cu_np, _, starts = layout_of(drafts_per_req)
    num_logits = int(cu_np[-1])
    target_logits = peak_logits([(7 * j + 3) % VOCAB for j in range(num_logits)], VOCAB)
    draft_logits = None
    draft_sampled = torch.zeros(num_logits, dtype=INT32)
    if with_draft_logits:
        draft_logits = torch.randn(MAX_NUM_REQS, k, VOCAB, generator=gen) * 2.0
    for r, start in enumerate(starts):
        slot = slots[r]
        draft_sampled[start] = 0  # 块外令牌，内核不读
        for i in range(k):
            if with_draft_logits:
                token = int(draft_logits[slot, i].argmax())
            else:
                token = (7 * (start + i) + 3) % VOCAB
            # 请求 2 的第 1 步留成 -1 占位（非贪心下它终止整块的验证）
            draft_sampled[start + i + 1] = -1 if (r == 2 and i == 1) else token
    inputs = build_v2_inputs(
        drafts_per_req=drafts_per_req, slots=slots, num_speculative_steps=k,
        target_logits=target_logits, draft_sampled=draft_sampled,
        temperature_by_slot={5: 1.0, 0: 0.5, 3: 2.0},
        seed_by_slot={5: 999983, 0: 12345, 3: 424242},
        draft_logits=draft_logits, use_block_verification=True,
        max_num_reqs=MAX_NUM_REQS)
    from spec75_helpers import assert_same_outputs

    assert_same_outputs(inputs, inputs.call(mv_rs), inputs.call(up_rs))


def test_block_matches_frozen_kernel_with_greedy_and_scrambled_slots() -> None:
    """混合批：贪心 + 非贪心 + 0 草稿请求 + 乱序 slot；贪心请求走 argmax 分支（块分支不参与）。"""
    drafts_per_req = [3, 0, 2, 1]
    slots = [6, 2, 5, 0]
    cu_np, _, starts = layout_of(drafts_per_req)
    num_logits = int(cu_np[-1])
    target_logits = peak_logits([(3 * j + 2) % VOCAB for j in range(num_logits)], VOCAB)
    draft_sampled = torch.zeros(num_logits, dtype=INT32)
    for r, start in enumerate(starts):
        for i in range(drafts_per_req[r]):
            offset = 2 if (r + i) % 2 == 0 else 3
            token = (3 * (start + i) + offset) % VOCAB
            draft_sampled[start + i + 1] = token
    inputs = build_v2_inputs(
        drafts_per_req=drafts_per_req, slots=slots, num_speculative_steps=3,
        target_logits=target_logits, draft_sampled=draft_sampled,
        temperature_by_slot={6: 0.0, 2: 1.0, 5: 0.0, 0: 1.5},
        seed_by_slot={6: 31337, 2: 2718, 5: 1414, 0: 1732},
        use_block_verification=True, max_num_reqs=MAX_NUM_REQS)
    from spec75_helpers import assert_same_outputs

    assert_same_outputs(inputs, inputs.call(mv_rs), inputs.call(up_rs))


# ---------------------------------------------------------------------------
# ② 逐个 kernel 的中间缓冲差分（定位到"哪一段移植错了"）
# ---------------------------------------------------------------------------


def _stagewise(module, inputs, *, zeros: bool = True):
    """按 `rejection_sample` 的顺序逐 kernel 跑一遍，返回全部中间缓冲。

    用 `zeros` 而不是 `new_empty` 分配：两边从同一份初值出发，"内核没写的格子"也能比较
    （否则比的是未初始化显存）。整体通路的差分（上面两个用例）用的是真分配，两者互补。
    """
    torch_mod = torch
    target = inputs.target_logits
    num_logits, vocab_size = target.shape
    num_reqs = inputs.num_reqs
    k = inputs.num_speculative_steps
    draft_logits = inputs.draft_logits
    has_draft = draft_logits is not None
    draft_stride_0 = draft_logits.stride(0) if has_draft else 0
    draft_stride_1 = draft_logits.stride(1) if has_draft else 0
    if has_draft:
        vocab_size = min(vocab_size, draft_logits.size(-1))

    block = 8192
    n_blocks = (vocab_size + block - 1) // block
    pad_blocks = 1 << (n_blocks - 1).bit_length()

    def alloc(shape, dtype):
        return torch_mod.zeros(shape, dtype=dtype, device=target.device)

    buf = {}
    buf["target_local_argmax"] = alloc((num_logits, n_blocks), torch_mod.int64)
    buf["target_local_max"] = alloc((num_logits, n_blocks), torch_mod.float32)
    buf["target_local_sumexp"] = alloc((num_logits, n_blocks), torch_mod.float32)
    buf["draft_local_max"] = alloc((num_logits, n_blocks), torch_mod.float32)
    buf["draft_local_sumexp"] = alloc((num_logits, n_blocks), torch_mod.float32)
    module._compute_local_logits_stats_kernel[(num_logits, n_blocks)](
        buf["target_local_argmax"], buf["target_local_argmax"].stride(0),
        buf["target_local_max"], buf["target_local_max"].stride(0),
        buf["target_local_sumexp"], buf["target_local_sumexp"].stride(0),
        buf["draft_local_max"], buf["draft_local_max"].stride(0),
        buf["draft_local_sumexp"], buf["draft_local_sumexp"].stride(0),
        target, target.stride(0), draft_logits, draft_stride_0, draft_stride_1,
        inputs.expanded_idx_mapping, inputs.expanded_local_pos, inputs.temperature,
        vocab_size, k, BLOCK_SIZE=block, HAS_DRAFT_LOGITS=has_draft)

    cumulative = alloc((num_logits,), torch_mod.float32)
    module._compute_cumulative_log_p_kernel[(num_reqs,)](
        cumulative, target, target.stride(0),
        buf["target_local_max"], buf["target_local_max"].stride(0),
        buf["target_local_sumexp"], buf["target_local_sumexp"].stride(0),
        inputs.draft_sampled, draft_logits, draft_stride_0, draft_stride_1,
        buf["draft_local_max"], buf["draft_local_max"].stride(0),
        buf["draft_local_sumexp"], buf["draft_local_sumexp"].stride(0),
        inputs.cu_num_logits, inputs.idx_mapping, inputs.temperature, n_blocks,
        PADDED_VOCAB_NUM_BLOCKS=pad_blocks, HAS_DRAFT_LOGITS=has_draft, num_warps=1)
    buf["cumulative_log_p"] = cumulative

    local_residual = None
    if has_draft:
        local_residual = alloc((num_logits, n_blocks), torch_mod.float32)
        module._compute_local_residual_mass_kernel[(num_logits, n_blocks)](
            local_residual, local_residual.stride(0), cumulative,
            target, target.stride(0),
            buf["target_local_max"], buf["target_local_max"].stride(0),
            buf["target_local_sumexp"], buf["target_local_sumexp"].stride(0),
            draft_logits, draft_stride_0, draft_stride_1,
            buf["draft_local_max"], buf["draft_local_max"].stride(0),
            buf["draft_local_sumexp"], buf["draft_local_sumexp"].stride(0),
            inputs.draft_sampled, inputs.expanded_idx_mapping, inputs.expanded_local_pos,
            inputs.temperature, vocab_size, k, n_blocks,
            BLOCK_SIZE=block, PADDED_VOCAB_NUM_BLOCKS=pad_blocks)
        buf["local_residual_mass"] = local_residual

    sampled = alloc((num_reqs, k + 1), torch_mod.int64)
    num_sampled = alloc((num_reqs,), torch_mod.int32)
    target_lse = alloc((num_reqs,), torch_mod.float32)
    draft_lse = alloc((num_reqs,), torch_mod.float32)
    module._rejection_kernel[(num_reqs,)](
        sampled, sampled.stride(0), num_sampled, target_lse, draft_lse,
        target, target.stride(0),
        buf["target_local_argmax"], buf["target_local_argmax"].stride(0),
        buf["target_local_max"], buf["target_local_max"].stride(0),
        buf["target_local_sumexp"], buf["target_local_sumexp"].stride(0),
        inputs.draft_sampled, draft_logits, draft_stride_0, draft_stride_1,
        buf["draft_local_max"], buf["draft_local_max"].stride(0),
        buf["draft_local_sumexp"], buf["draft_local_sumexp"].stride(0),
        inputs.cu_num_logits, inputs.idx_mapping, inputs.temperature, inputs.seeds,
        inputs.pos, inputs.synthetic_conditional_rates, cumulative,
        local_residual, local_residual.stride(0) if local_residual is not None else 0,
        n_blocks, PADDED_VOCAB_NUM_BLOCKS=pad_blocks, HAS_DRAFT_LOGITS=has_draft,
        SYNTHETIC_MODE=inputs.synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=inputs.use_block_verification, num_warps=1)
    buf["sampled"] = sampled
    buf["num_sampled"] = num_sampled
    buf["target_rejected_logsumexp"] = target_lse
    buf["draft_rejected_logsumexp"] = draft_lse

    resample_block = 1024
    resample_blocks = (vocab_size + resample_block - 1) // resample_block
    padded_resample = 1 << (resample_blocks - 1).bit_length()
    local_argmax = alloc((num_reqs, resample_blocks), torch_mod.int64)
    local_max = alloc((num_reqs, resample_blocks), torch_mod.float32)
    module._resample_kernel[(num_reqs, resample_blocks)](
        local_argmax, local_argmax.stride(0), local_max, local_max.stride(0),
        target, target.stride(0), target_lse,
        draft_logits, draft_stride_0, draft_stride_1, draft_lse,
        num_sampled, inputs.cu_num_logits, inputs.expanded_idx_mapping,
        inputs.draft_sampled, inputs.temperature, inputs.seeds, inputs.pos,
        cumulative, vocab_size, BLOCK_SIZE=resample_block, HAS_DRAFT_LOGITS=has_draft,
        USE_FP64=inputs.use_fp64, USE_BLOCK_VERIFICATION=inputs.use_block_verification)
    buf["resampled_local_argmax"] = local_argmax
    buf["resampled_local_max"] = local_max

    module._insert_resampled_kernel[(num_reqs,)](
        sampled, sampled.stride(0), num_sampled,
        local_argmax, local_argmax.stride(0), local_max, local_max.stride(0),
        resample_blocks, inputs.cu_num_logits, inputs.expanded_idx_mapping,
        inputs.temperature, PADDED_RESAMPLE_NUM_BLOCKS=padded_resample)
    return buf


@pytest.mark.parametrize("with_draft_logits", [False, True])
def test_each_block_kernel_matches_frozen_stage_by_stage(with_draft_logits: bool) -> None:
    """K=3、混合温度：逐 kernel 比较中间缓冲（累计对数比 / 残差质量 / logsumexp / 局部 argmax）。"""
    k = 3
    drafts_per_req = [3, 3, 3]
    slots = [4, 1, 6]
    cu_np, _, starts = layout_of(drafts_per_req)
    num_logits = int(cu_np[-1])
    gen = torch.Generator().manual_seed(7500)
    target_logits = torch.randn(num_logits, VOCAB, generator=gen) * 2.0
    draft_logits = None
    draft_sampled = torch.zeros(num_logits, dtype=INT32)
    if with_draft_logits:
        draft_logits = torch.randn(MAX_NUM_REQS, k, VOCAB, generator=gen) * 2.0
    for r, start in enumerate(starts):
        for i in range(k):
            token = int(draft_logits[slots[r], i].argmax()) if with_draft_logits \
                else int(target_logits[start + i].argmax())
            draft_sampled[start + i + 1] = token
    inputs = build_v2_inputs(
        drafts_per_req=drafts_per_req, slots=slots, num_speculative_steps=k,
        target_logits=target_logits, draft_sampled=draft_sampled,
        temperature_by_slot={4: 0.5, 1: 1.0, 6: 2.0},
        seed_by_slot={4: 11, 1: 22, 6: 33},
        draft_logits=draft_logits, use_block_verification=True, max_num_reqs=MAX_NUM_REQS)

    mine = _stagewise(mv_rs, inputs)
    frozen = _stagewise(up_rs, inputs)
    assert set(mine) == set(frozen)
    for name in mine:
        a, b = mine[name], frozen[name]
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert torch.equal(a, b), f"{name} 不一致（逐 kernel 差分失败）"


# ---------------------------------------------------------------------------
# ③ 分布对照：Torch 数学参考 vs 内核（比分布，不比逐值）
# ---------------------------------------------------------------------------


def test_block_acceptance_length_distribution_matches_reference() -> None:
    """τ 的分布（τ = num_sampled - 1）与参考的精确分布一致。

    参考的 τ 分布是闭式：`P(τ=n) = h_{n-1} · Π_{j>=n}(1 - h_j)`（块验证每位都掷硬币）。
    """
    case = residual_case()
    n_trials = 8000
    _sampled, num_sampled = run_trials(case, n_trials)
    empirical = empirical_acceptance_distribution(num_sampled, case.num_drafts)
    expected = case.acceptance_length_distribution(block=True)
    assert total_variation(empirical, expected) < 0.02, (
        f"τ 分布不一致：经验 {empirical.tolist()} vs 参考 {expected.tolist()}")
    mean = float((num_sampled.to(torch.float64) - 1).mean())
    assert abs(mean - case.mean_acceptance_length(block=True)) < 0.05


def test_block_residual_distribution_matches_reference_not_standard() -> None:
    """τ=1 时**交付的那枚 token** 的分布必须匹配块残差，而不是 standard 的残差。

    这是"不能只比接受率"的那条：两种验证方式在 τ=1 上的接受概率可能相同，但恢复分布不同
    （块残差带 `p_tau` 缩放）。本用例实测 TV(经验, 块参考) ≈ 0.007，而 TV(块参考, standard 参考)
    = 0.19 —— 所以匹配块参考是有判别力的，写成 standard 会被卡方抓住。
    """
    case = residual_case()
    n_trials = 8000
    sampled, num_sampled = run_trials(case, n_trials)
    histogram = conditional_token_histogram(sampled, num_sampled, tau_wanted=1)

    block_ref = case.residual_probs(1, block=True)
    standard_ref = case.residual_probs(1, block=False)
    # 先证明这两个参考真的不同（否则本用例是恒真的）
    assert total_variation(block_ref, standard_ref) > 0.15

    assert total_variation(histogram, block_ref) < 0.03, (
        f"τ=1 的交付分布没匹配块残差：经验 {histogram.tolist()} vs 参考 {block_ref.tolist()}")
    assert chi_square(histogram, block_ref) < 30.0
    assert total_variation(histogram, standard_ref) > 0.10


def test_block_one_hot_draft_closed_form_matches_reference() -> None:
    """点质量草稿：τ 分布与**交付 token 的混合分布**都与闭式参考一致（四个 τ 分支都覆盖）。"""
    case = one_hot_case()
    n_trials = 8000
    sampled, num_sampled = run_trials(case, n_trials)
    empirical = empirical_acceptance_distribution(num_sampled, case.num_drafts)
    expected = case.acceptance_length_distribution(block=True)
    assert total_variation(empirical, expected) < 0.02, (
        f"τ 分布不一致：{empirical.tolist()} vs {expected.tolist()}")

    # 交付 token 的**混合分布**（对所有 τ 加权）——它同时约束四个 τ 下的残差公式
    rows = torch.arange(n_trials, device=sampled.device)
    tokens = sampled[rows, num_sampled.to(torch.int64) - 1].to(torch.int64).cpu()
    histogram = torch.bincount(tokens, minlength=VOCAB).double() / n_trials
    assert total_variation(histogram, case.delivered_distribution(block=True)) < 0.03, (
        f"交付分布不一致：{histogram.tolist()} vs {case.delivered_distribution(block=True).tolist()}")


def test_block_bonus_branch_and_last_candidate() -> None:
    """K=1：最后一位的阈值就是 `p_i`（没有下一行），全接受时 bonus 从 bonus 行的 target 采。"""
    case = bonus_case()
    n_trials = 8000
    sampled, num_sampled = run_trials(case, n_trials)
    empirical = empirical_acceptance_distribution(num_sampled, case.num_drafts)
    assert total_variation(empirical, case.acceptance_length_distribution(True)) < 0.02

    rows = torch.arange(n_trials, device=sampled.device)
    tokens = sampled[rows, num_sampled.to(torch.int64) - 1].to(torch.int64).cpu()
    histogram = torch.bincount(tokens, minlength=VOCAB).double() / n_trials
    assert total_variation(histogram, case.delivered_distribution(True)) < 0.03

    # 两个分支都真的出现过（否则"混合分布匹配"可能只是匹配了其中一个分支）
    tau = (num_sampled.to(torch.int64) - 1)
    assert int((tau == 0).sum()) > 1000 and int((tau == 1).sum()) > 1000


def test_block_all_accepts_when_draft_equals_target() -> None:
    """`p == q`（草稿分布逐位等于 target）→ 阈值 0/0 走兜底 1.0 → 全接受（数值稳定性）。"""
    case = perfect_case(k=3)
    assert all(float(h) == 1.0 for h in case.thresholds())
    _sampled, num_sampled = run_trials(case, 512)
    assert bool((num_sampled == case.num_drafts + 1).all()), num_sampled


def test_block_zero_probability_draft_is_rejected_at_first_position() -> None:
    """首接受长度为 0：`p_0 = 0` → `h_0 = 0` → 必拒；此时块残差退化成 standard 的形状。"""
    case = zero_prob_case()
    assert float(case.prefix_joint_ratio()[0]) == 0.0
    assert float(case.thresholds()[0]) == 0.0
    n_trials = 8000
    sampled, num_sampled = run_trials(case, n_trials)
    assert bool((num_sampled == 1).all()), num_sampled

    # τ=0 → `p_tau = 1`，于是块残差 == standard 残差（都等于"target 去掉被拒草稿 token"）
    assert torch.allclose(case.residual_probs(0, True), case.residual_probs(0, False))
    rows = torch.arange(n_trials, device=sampled.device)
    tokens = sampled[rows, num_sampled.to(torch.int64) - 1].to(torch.int64).cpu()
    histogram = torch.bincount(tokens, minlength=VOCAB).double() / n_trials
    assert total_variation(histogram, case.residual_probs(0, True)) < 0.05
    # 被拒的草稿 token 5 永远不能出现在交付里
    assert int(histogram[5]) == 0


def test_block_and_standard_disagree_on_the_same_input() -> None:
    """同一批输入：block 与 standard 的 τ 分布必须不同（否则"块分支"根本没被走到）。"""
    case = residual_case()
    n_trials = 4000
    _s_blk, n_blk = run_trials(case, n_trials, use_std=False)
    _s_std, n_std = run_trials(case, n_trials, use_std=True)
    emp_blk = empirical_acceptance_distribution(n_blk, case.num_drafts)
    emp_std = empirical_acceptance_distribution(n_std, case.num_drafts)
    ref_blk = case.acceptance_length_distribution(True)
    ref_std = case.acceptance_length_distribution(False)
    assert total_variation(emp_blk, ref_blk) < 0.03
    assert total_variation(emp_std, ref_std) < 0.03
    # 两条参考本身差得足够远（本设计：E[τ] 0.366 vs 0.637）
    assert total_variation(ref_blk, ref_std) > 0.15
    # 反向：故意拿错参考去比，必须报错（证明上面的阈值不是"怎么都过"）
    assert total_variation(emp_blk, ref_std) > 0.10


def test_block_greedy_requests_take_the_argmax_path() -> None:
    """贪心请求在块验证下仍走 `is_greedy` 分支：块阈值一个都不参与，结果 == standard。"""
    k = 2
    drafts_per_req = [k, k]
    slots = [0, 1]
    cu_np, _, starts = layout_of(drafts_per_req)
    num_logits = int(cu_np[-1])
    row_tokens = [(5 * j + 1) % VOCAB for j in range(num_logits)]
    target_logits = peak_logits(row_tokens, VOCAB)
    draft_sampled = torch.zeros(num_logits, dtype=INT32)
    for r, start in enumerate(starts):
        # 请求 0：草稿 = argmax（全接受）；请求 1：第 0 步错开（拒绝）
        draft_sampled[start + 1] = row_tokens[start] if r == 0 else (row_tokens[start] + 1) % VOCAB
        draft_sampled[start + 2] = row_tokens[start + 1]
    common = dict(
        drafts_per_req=drafts_per_req, slots=slots, num_speculative_steps=k,
        target_logits=target_logits, draft_sampled=draft_sampled,
        temperature_by_slot={0: 0.0, 1: 0.0}, seed_by_slot={0: 7, 1: 9},
        max_num_reqs=MAX_NUM_REQS)
    blk = build_v2_inputs(**common, use_block_verification=True).call(mv_rs)
    std = build_v2_inputs(**common, use_block_verification=False).call(mv_rs)
    assert torch.equal(blk[1], std[1])
    assert blk[1].tolist() == [k + 1, 1]
    # 语义钉死（证明"逐值相等"不是恒真）：请求 0 草稿全中 → K+1 枚；请求 1 第 0 步就错开 → 1 枚
    # （请求 1 的行块是 3..5：第 0 步写在行 3，被拒时顶替成该行的 target argmax = row_tokens[3]）
    assert blk[0][0, :k + 1].tolist() == row_tokens[0:k + 1]
    assert blk[0][1, 0].item() == row_tokens[3]


def test_block_rejects_synthetic_rates_explicitly() -> None:
    """block 与 synthetic 互斥：同时给会被 assert 挡住（上游同款）。"""
    inputs = build_v2_inputs(
        drafts_per_req=[1], slots=[0], num_speculative_steps=1,
        target_logits=peak_logits([0, 0], VOCAB), draft_sampled=torch.tensor([0, 0], dtype=INT32),
        temperature_by_slot={0: 1.0}, seed_by_slot={0: 1},
        synthetic_conditional_rates=torch.tensor([0.5], dtype=torch.float32),
        use_block_verification=True, max_num_reqs=MAX_NUM_REQS)
    with pytest.raises(AssertionError, match="incompatible"):
        inputs.call(mv_rs)
