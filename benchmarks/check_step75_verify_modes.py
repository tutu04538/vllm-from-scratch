"""75 关验收探针：V2 块验证 + synthetic 接受率 + 分块 logits（跑法见文末）。

跑法：`python benchmarks/check_step75_verify_modes.py`（退出码非 0 = 失败）。

分段（对应需求 075 §3/§4）：

    A 配置与换算   A、unconditional→conditional（需求原例）、length→rates、与上游逐项同判、
                   V1/V2 采样器的模式分派（含 V1 对 block 的明确报错）
    B 逐 kernel 差分 块验证的五个内核（stats / cumulative_log_p / residual_mass / rejection /
                   resample / insert）与冻结实现逐值一致
    C 分布对照      block 的 τ 分布与 τ=1 残差分布 vs Torch 数学参考（TV / 卡方），
                   并给出"若写成 standard 残差会被抓住"的判别力数字
    D E[τ] 对照     block vs standard 随草稿质量（mix）的变化——论文区间 vs 对抗性合成区间
    E synthetic     经验前缀接受率 == 配置的无条件率（8000 次）；全 0/全 1 边界
    F 无损性反证    rates=1 时接受一枚 target 概率仅 ~3.3e-4 的草稿 token（synthetic 不保证无损）
    G 分块          单块 vs 多块逐值一致；块数与上限的实测；陈旧 CPU offsets 的保护
    H V1 回归       V1 synthetic 与上游逐值差分（钉死随机源）；V1/V2 的 τ 分布一致

⚠️ 真实权重的接受长度受 63 关已知差距影响（`docs/step63_alignment.md` §8），本探针只用
synthetic/合成分布，验的是**协议与数学**，不是草稿质量。
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step75"))

from spec75_helpers import (DEVICE, MAX_NUM_REQS, VOCAB, layout_of,  # noqa: E402
                            make_metadata, patch_random_sources, run_upstream_v1,
                            run_v1, sampling_metadata)

import minivllm.worker.gpu.spec_decode.rejection_sampler_utils as mv_rs  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as up_rs  # noqa: E402
from minivllm.config import SpeculativeConfig  # noqa: E402
from minivllm.sample import Sampler as V1Sampler  # noqa: E402
from minivllm.sample.rejection_sampler import RejectionSampler as V1RejectionSampler  # noqa: E402
from minivllm.spec_decode.utils import unconditional_to_conditional_rates  # noqa: E402
from minivllm.testing.torch_block_verification import (BlockCase,  # noqa: E402
                                                       total_variation)
from minivllm.worker.gpu.spec_decode.rejection_sampler import (  # noqa: E402
    RejectionSampler as V2RejectionSampler,
)

PASSED, FAILED, TRACE = 0, [], []


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}  {detail}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")
    TRACE.append({"item": name, "ok": bool(ok), "detail": str(detail)[:300]})


if not torch.cuda.is_available():  # pragma: no cover - 本机有 CUDA
    print("SKIP  探针需要 CUDA（拒绝采样是 Triton 内核）")
    raise SystemExit(0)

RATES = [0.8, 0.4, 0.2]
#: 块验证那几段用的小词表（与 `BlockCase` 的手算数字配套；tiny 引擎的词表是 `VOCAB`）
VB = 8
CONDITIONAL = [0.8, 0.5, 0.5]
K = 3

# ================================================ A. 配置与换算

conv = unconditional_to_conditional_rates(RATES)
check("A1. 无条件→条件换算（需求 075 §3.4 原例 [0.8,0.4,0.2] → [0.8,0.5,0.5]）",
      conv == CONDITIONAL, conv)

from vllm.v1.spec_decode.utils import (  # noqa: E402
    unconditional_to_conditional_rates as upstream_conv,
)
same = all(unconditional_to_conditional_rates(list(r)) == upstream_conv(list(r)) for r in
           ([0.8, 0.4, 0.2], [1.0], [0.0], [0.0, 0.0], [0.9, 0.8, 0.7, 0.0]))
check("A2. 与上游同名函数逐值一致（含 q==0 退化分支）", same, conv)

from vllm.config.speculative import SpeculativeConfig as UpstreamSpec  # noqa: E402

cases = [(3, [0.8, 0.4, 0.2], None), (3, [0.8, 0.4], None), (3, [0.8, 0.9, 0.2], None),
         (3, [1.2, 0.4, 0.2], None), (3, None, None), (3, [0.8, 0.4, 0.2], 2.0),
         (3, None, 4.5), (4, None, 2.5), (1, None, 2.0)]
agree = True
for n, rates, length in cases:
    mine = upstream = None
    try:
        mine = SpeculativeConfig._resolve_synthetic_acceptance_rates(n, rates, length)
    except ValueError as exc:
        mine = f"ERR:{exc}"
    try:
        upstream = UpstreamSpec._resolve_synthetic_acceptance_rates(n, rates, length)
    except ValueError as exc:
        upstream = f"ERR:{exc}"
    agree &= mine == upstream
check("A3. 长度/单调/范围/二选一的校验与上游逐项同判（连错误原文都同）", agree, f"{len(cases)} 例")

spec_len = SpeculativeConfig(method="ngram", num_speculative_tokens=4,
                             rejection_sample_method="synthetic",
                             synthetic_acceptance_length=2.5)
check("A4. `synthetic_acceptance_length=2.5`(K=4) → rates [1,0.5,0,0] 且字段被清空",
      spec_len.synthetic_acceptance_rates == [1.0, 0.5, 0.0, 0.0]
      and spec_len.synthetic_acceptance_length is None,
      spec_len.synthetic_acceptance_rates)


class _StubSampler:
    logprobs_mode = "raw_logprobs"


spec_syn = SpeculativeConfig(method="ngram", num_speculative_tokens=K,
                            rejection_sample_method="synthetic",
                            synthetic_acceptance_rates=RATES)
spec_blk = SpeculativeConfig(method="ngram", num_speculative_tokens=K,
                            rejection_sample_method="block")
v2_syn = V2RejectionSampler(_StubSampler(), spec_syn, DEVICE)
v2_blk = V2RejectionSampler(_StubSampler(), spec_blk, DEVICE)
v1_syn = V1RejectionSampler(V1Sampler("raw_logprobs"), spec_syn, DEVICE)
v1_block_error = None
try:
    V1RejectionSampler(V1Sampler("raw_logprobs"), spec_blk, DEVICE)
except NotImplementedError as exc:
    v1_block_error = str(exc)
check("A5. 模式分派：V2 synthetic 建条件率表 / block 置布尔；V1 对 block 明确报错（不静默按 standard）",
      np.allclose(v2_syn.synthetic_conditional_rates.tolist(), CONDITIONAL, atol=1e-6)
      and v2_blk.use_block_verification is True and v2_blk.synthetic_conditional_rates is None
      and v1_syn.synthetic_mode is True and v1_block_error is not None and "V2" in v1_block_error,
      f"V2 条件率={v2_syn.synthetic_conditional_rates.tolist()}、"
      f"V1 block 报错={'ok' if v1_block_error else 'NO'}")

# ================================================ 工具：一次调用塞 N 条请求（每请求一个 seed）


def run_trials(case: BlockCase, n_trials: int, *, module=mv_rs, use_std=False,
               seed_base=0, pos_start=100):
    rows = case.num_drafts + 1
    target = case.target_logits.to(DEVICE).repeat(n_trials, 1).contiguous()
    draft_logits = None
    if case.draft_logits is not None:
        draft_logits = torch.zeros(n_trials, case.num_drafts, VB,
                                   dtype=torch.float32, device=DEVICE)
        draft_logits[:] = case.draft_logits.to(DEVICE)
    draft_sampled = torch.zeros(n_trials * rows, dtype=torch.int32, device=DEVICE)
    for i, token in enumerate(case.draft_tokens):
        draft_sampled[i + 1::rows] = token
    cu = torch.arange(0, (n_trials + 1) * rows, rows, dtype=torch.int32, device=DEVICE)
    pos = torch.arange(0, n_trials * rows, dtype=torch.int64, device=DEVICE) + pos_start
    idx = torch.arange(n_trials, dtype=torch.int64, device=DEVICE)
    return module.rejection_sample(
        target, draft_logits, draft_sampled, cu, pos, idx, idx.repeat_interleave(rows),
        torch.arange(rows, dtype=torch.int32, device=DEVICE).repeat(n_trials),
        torch.full((n_trials,), case.temperature, dtype=torch.float32, device=DEVICE),
        torch.arange(n_trials, dtype=torch.int64, device=DEVICE) + seed_base,
        case.num_drafts, None, use_fp64=False, use_block_verification=not use_std)


def tau_distribution(num_sampled, k):
    tau = (num_sampled.to(torch.int64) - 1).cpu()
    return torch.bincount(tau, minlength=k + 1).double() / num_sampled.numel()


# ================================================ B. 逐 kernel 差分（块验证的五个内核）

def stagewise(module, target, draft_logits, draft_sampled, cu, pos, idx, exp_idx, exp_pos,
              temp, seeds, k):
    """按 `rejection_sample` 的顺序逐 kernel 跑一遍（zeros 分配 → 未写的格子也可比）。"""
    num_logits, vocab_size = target.shape
    num_reqs = cu.numel() - 1
    has_draft = draft_logits is not None
    s0 = draft_logits.stride(0) if has_draft else 0
    s1 = draft_logits.stride(1) if has_draft else 0
    if has_draft:
        vocab_size = min(vocab_size, draft_logits.size(-1))
    block = 8192
    n_blocks = (vocab_size + block - 1) // block
    pad = 1 << (n_blocks - 1).bit_length()
    z = lambda *shape: torch.zeros(*shape, device=target.device, dtype=torch.float32)  # noqa: E731
    t_argmax = torch.zeros((num_logits, n_blocks), dtype=torch.int64, device=target.device)
    t_max, t_sx = z(num_logits, n_blocks), z(num_logits, n_blocks)
    d_max, d_sx = z(num_logits, n_blocks), z(num_logits, n_blocks)
    module._compute_local_logits_stats_kernel[(num_logits, n_blocks)](
        t_argmax, t_argmax.stride(0), t_max, t_max.stride(0), t_sx, t_sx.stride(0),
        d_max, d_max.stride(0), d_sx, d_sx.stride(0), target, target.stride(0),
        draft_logits, s0, s1, exp_idx, exp_pos, temp, vocab_size, k,
        BLOCK_SIZE=block, HAS_DRAFT_LOGITS=has_draft)
    cum = z(num_logits)
    module._compute_cumulative_log_p_kernel[(num_reqs,)](
        cum, target, target.stride(0), t_max, t_max.stride(0), t_sx, t_sx.stride(0),
        draft_sampled, draft_logits, s0, s1, d_max, d_max.stride(0), d_sx, d_sx.stride(0),
        cu, idx, temp, n_blocks, PADDED_VOCAB_NUM_BLOCKS=pad,
        HAS_DRAFT_LOGITS=has_draft, num_warps=1)
    local_res = None
    if has_draft:
        local_res = z(num_logits, n_blocks)
        module._compute_local_residual_mass_kernel[(num_logits, n_blocks)](
            local_res, local_res.stride(0), cum, target, target.stride(0),
            t_max, t_max.stride(0), t_sx, t_sx.stride(0), draft_logits, s0, s1,
            d_max, d_max.stride(0), d_sx, d_sx.stride(0), draft_sampled, exp_idx, exp_pos,
            temp, vocab_size, k, n_blocks, BLOCK_SIZE=block, PADDED_VOCAB_NUM_BLOCKS=pad)
    sampled = torch.zeros((num_reqs, k + 1), dtype=torch.int64, device=target.device)
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=target.device)
    t_lse, d_lse = z(num_reqs), z(num_reqs)
    module._rejection_kernel[(num_reqs,)](
        sampled, sampled.stride(0), num_sampled, t_lse, d_lse, target, target.stride(0),
        t_argmax, t_argmax.stride(0), t_max, t_max.stride(0), t_sx, t_sx.stride(0),
        draft_sampled, draft_logits, s0, s1, d_max, d_max.stride(0), d_sx, d_sx.stride(0),
        cu, idx, temp, seeds, pos, None, cum,
        local_res, local_res.stride(0) if local_res is not None else 0, n_blocks,
        PADDED_VOCAB_NUM_BLOCKS=pad, HAS_DRAFT_LOGITS=has_draft, SYNTHETIC_MODE=False,
        USE_BLOCK_VERIFICATION=True, num_warps=1)
    rb = 1024
    n_rb = (vocab_size + rb - 1) // rb
    loc_arg = torch.zeros((num_reqs, n_rb), dtype=torch.int64, device=target.device)
    loc_max = z(num_reqs, n_rb)
    module._resample_kernel[(num_reqs, n_rb)](
        loc_arg, loc_arg.stride(0), loc_max, loc_max.stride(0), target, target.stride(0),
        t_lse, draft_logits, s0, s1, d_lse, num_sampled, cu, exp_idx, draft_sampled, temp,
        seeds, pos, cum, vocab_size, BLOCK_SIZE=rb, HAS_DRAFT_LOGITS=has_draft,
        USE_FP64=False, USE_BLOCK_VERIFICATION=True)
    module._insert_resampled_kernel[(num_reqs,)](
        sampled, sampled.stride(0), num_sampled, loc_arg, loc_arg.stride(0),
        loc_max, loc_max.stride(0), n_rb, cu, exp_idx, temp,
        PADDED_RESAMPLE_NUM_BLOCKS=1 << (n_rb - 1).bit_length())
    return {"cumulative_log_p": cum, "local_residual_mass": local_res,
            "target_local_argmax": t_argmax, "target_local_max": t_max,
            "target_local_sumexp": t_sx, "target_rejected_lse": t_lse,
            "draft_rejected_lse": d_lse, "resampled_local_argmax": loc_arg,
            "resampled_local_max": loc_max, "sampled": sampled, "num_sampled": num_sampled}


for with_draft in (False, True):
    k, drafts_per_req, slots = 3, [3, 3, 3], [4, 1, 6]
    cu_np, local_pos_np, starts = layout_of(drafts_per_req)
    num_logits = int(cu_np[-1])
    gen = torch.Generator().manual_seed(7500)
    target = (torch.randn(num_logits, VOCAB, generator=gen) * 2.0).to(DEVICE)
    draft_logits = None
    draft_sampled = torch.zeros(num_logits, dtype=torch.int32, device=DEVICE)
    if with_draft:
        draft_logits = (torch.randn(MAX_NUM_REQS, k, VOCAB, generator=gen) * 2.0).to(DEVICE)
    cpu_tokens = torch.zeros(num_logits, dtype=torch.int32)
    for r, start in enumerate(starts):
        for i in range(k):
            token = int(draft_logits[slots[r], i].argmax()) if with_draft \
                else int(target[start + i].argmax())
            cpu_tokens[start + i + 1] = token
    draft_sampled = cpu_tokens.to(DEVICE)
    cu = torch.from_numpy(cu_np).to(DEVICE)
    pos = (torch.arange(num_logits, dtype=torch.int64, device=DEVICE) + 100)
    idx = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    exp_idx = idx.repeat_interleave(k + 1)
    exp_pos = torch.from_numpy(local_pos_np).to(DEVICE)
    temp = torch.zeros(MAX_NUM_REQS, dtype=torch.float32, device=DEVICE)
    seeds = torch.zeros(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE)
    for i, slot in enumerate(slots):
        temp[slot] = (0.5, 1.0, 2.0)[i]
        seeds[slot] = 11 * (i + 1)
    mine = stagewise(mv_rs, target, draft_logits, draft_sampled, cu, pos, idx, exp_idx, exp_pos,
                     temp, seeds, k)
    frozen = stagewise(up_rs, target, draft_logits, draft_sampled, cu, pos, idx, exp_idx,
                       exp_pos, temp, seeds, k)
    diffs = [name for name, value in mine.items()
             if (value is None) != (frozen[name] is None)
             or (value is not None and not torch.equal(value, frozen[name]))]
    check(f"B1. 块验证逐 kernel 差分（点质量草稿={not with_draft}）："
          f"5 个内核 × 11 个中间缓冲全部逐值相等",
          not diffs, f"不一致={diffs or '无'}")

# ================================================ C. 分布对照（Torch 数学参考）

residual_case = BlockCase(
    torch.tensor([[0.0] * VB,
                  [1.0, 0.5, 0.0, -1.0, 0.0, 0.5, 1.5, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    torch.tensor([[0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    [1, 3], 1.0)
N = 8000
sampled, num_sampled = run_trials(residual_case, N)
emp_tau = tau_distribution(num_sampled, residual_case.num_drafts)
ref_tau = residual_case.acceptance_length_distribution(True)
tv_tau = total_variation(emp_tau, ref_tau)
check(f"C1. block 的 τ 分布 == 参考闭式（N={N}）", tv_tau < 0.02,
      f"TV={tv_tau:.4f} 经验={[round(float(x), 4) for x in emp_tau]} "
      f"参考={[round(float(x), 4) for x in ref_tau]}")

taus = (num_sampled.to(torch.int64) - 1)
mask = taus == 1
rows = torch.arange(N, device=DEVICE)[mask]
tokens = sampled[rows, taus[mask]].to(torch.int64).cpu()
hist = torch.bincount(tokens, minlength=VB).double() / int(mask.sum())
ref_block = residual_case.residual_probs(1, block=True)
ref_std = residual_case.residual_probs(1, block=False)
tv_block = total_variation(hist, ref_block)
tv_std = total_variation(hist, ref_std)
chi2 = float(((hist - ref_block) ** 2 / ref_block.clamp_min(1e-12)).sum() * int(mask.sum()))
check(f"C2. τ=1 的交付 token 分布 == **块残差** max(p_tau·M_b − M_s,0)（n={int(mask.sum())}）",
      tv_block < 0.03 and chi2 < 30,
      f"TV vs 块残差={tv_block:.4f}（卡方={chi2:.2f}） vs standard 残差={tv_std:.4f}"
      f"、两条参考之间 TV={total_variation(ref_block, ref_std):.4f}")

# ================================================ D. E[τ] 对照（论文区间 vs 对抗性区间）

sweep = []
gen = torch.Generator().manual_seed(5)
for mix in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
    eb = es = 0.0
    n = 200
    for _ in range(n):
        tgt = torch.randn(K + 1, VB, generator=gen) * 2.0
        noise = torch.randn(K, VB, generator=gen) * 2.0
        drf = (1 - mix) * noise + mix * tgt[:K]
        case = BlockCase(tgt, drf, [int(drf[i].argmax()) for i in range(K)], 1.0)
        eb += case.mean_acceptance_length(True)
        es += case.mean_acceptance_length(False)
    sweep.append((mix, eb / n, es / n))
check("D1. E[τ] 随草稿质量：#参数正确（草稿≈target 时 block 不差于 standard；差草稿时更差）",
      all(eb >= es - 0.02 for _, eb, es in sweep if _ >= 0.5),
      " / ".join(f"mix={m:.2f}: block {eb:.3f} vs std {es:.3f}" for m, eb, es in sweep))

# ================================================ E. synthetic：经验前缀接受率


def synthetic_trials(cond_rates, n_trials, k, temperature=1.0):
    rows = k + 1
    gen = torch.Generator().manual_seed(4242)
    target = (torch.randn(n_trials * rows, VOCAB, generator=gen) * 1.5).to(DEVICE)
    draft_sampled = torch.zeros(n_trials * rows, dtype=torch.int32, device=DEVICE)
    draft_sampled[1::rows] = torch.randint(0, VOCAB, (n_trials,), generator=gen).to(DEVICE)
    cu = torch.arange(0, (n_trials + 1) * rows, rows, dtype=torch.int32, device=DEVICE)
    idx = torch.arange(n_trials, dtype=torch.int64, device=DEVICE)
    return mv_rs.rejection_sample(
        target, None, draft_sampled, cu,
        torch.arange(n_trials * rows, dtype=torch.int64, device=DEVICE) + 100,
        idx, idx.repeat_interleave(rows),
        torch.arange(rows, dtype=torch.int32, device=DEVICE).repeat(n_trials),
        torch.full((n_trials,), temperature, dtype=torch.float32, device=DEVICE),
        torch.arange(n_trials, dtype=torch.int64, device=DEVICE), k,
        torch.tensor(cond_rates, dtype=torch.float32, device=DEVICE))


_s, n_syn = synthetic_trials(CONDITIONAL, 8000, K)
tau_syn = n_syn.to(torch.int64) - 1
measured = [float((tau_syn >= i + 1).double().mean()) for i in range(K)]
ok_e = all(abs(m - r) < 0.025 for m, r in zip(measured, RATES))
check("E1. 经验前缀接受率 == 配置的无条件率（N=8000，容差 0.025 ≈ 4.5σ）", ok_e,
      f"实测 {[round(m, 4) for m in measured]} vs 配置 {RATES}；"
      f"平均接受枚数 {float(tau_syn.double().mean()):.4f} vs Σrates={sum(RATES)}")
wrong = [RATES[0], RATES[0] * RATES[1], RATES[0] * RATES[1] * RATES[2]]
check("E2. 判别力：把无条件数组当条件率用（错误口径）会得到 "
      f"{[round(x, 3) for x in wrong]}，与配置差 > 容差",
      all(abs(w - r) > 0.025 for w, r in zip(wrong[1:], RATES[1:])),
      f"错误口径 vs 配置：{[round(abs(w - r), 4) for w, r in zip(wrong, RATES)]}")

_s1, n_all = synthetic_trials([1.0] * K, 512, K)
_s0, n_none = synthetic_trials([0.0] * K, 512, K)
check("E3. 边界：rates 全 1 → 全接受（τ=K）；全 0 → 必拒（u>0，τ=0）",
      bool((n_all == K + 1).all()) and bool((n_none == 1).all()),
      f"全1: num_sampled={sorted(set(n_all.tolist()))}、全0: {sorted(set(n_none.tolist()))}")

# ================================================ F. 无损性反证


def lossless_probe():
    k, n = 2, 256
    rows = k + 1
    cu = torch.arange(0, (n + 1) * rows, rows, dtype=torch.int32, device=DEVICE)
    idx = torch.arange(n, dtype=torch.int64, device=DEVICE)
    target = torch.zeros(n * rows, VOCAB, dtype=torch.float32, device=DEVICE)
    target[:, 0] = 8.0
    draft_sampled = torch.zeros(n * rows, dtype=torch.int32, device=DEVICE)
    draft_sampled[1::rows] = 1
    draft_sampled[2::rows] = 2
    sampled, num_sampled = mv_rs.rejection_sample(
        target, None, draft_sampled, cu,
        torch.arange(n * rows, dtype=torch.int64, device=DEVICE) + 100,
        idx, idx.repeat_interleave(rows),
        torch.arange(rows, dtype=torch.int32, device=DEVICE).repeat(n),
        torch.ones(n, dtype=torch.float32, device=DEVICE),
        torch.arange(n, dtype=torch.int64, device=DEVICE), k,
        torch.ones(k, dtype=torch.float32, device=DEVICE))
    p1 = float(torch.softmax(target[0], dim=-1)[1])
    return p1, sampled, num_sampled


p_rare, sampled_rare, num_rare = lossless_probe()
check("F1. 无损性反证：rates=1 时照收 target 概率 ~3.3e-4 的草稿 token（synthetic 不保证分布无损）",
      p_rare < 4e-4 and bool((num_rare == 3).all()) and bool((sampled_rare[:, 0] == 1).all()),
      f"target p(token1)={p_rare:.3e}、仍被接受={bool((sampled_rare[:, 0] == 1).all())}")

# ================================================ G. 分块

from minivllm.worker.gpu.spec_decode.rejection_sampler import (  # noqa: E402
    MAX_CHUNK_BYTES, _iter_request_chunks, get_max_chunk_logits,
)

check("G1. 上限口径 = 1 GiB 的 fp32 行（151936→1766、32000→8388、128256→2092）",
      (get_max_chunk_logits(151936), get_max_chunk_logits(32000),
       get_max_chunk_logits(128256)) == (1766, 8388, 2092),
      f"MAX_CHUNK_BYTES={MAX_CHUNK_BYTES}")

#: 三条请求、各 K 枚草稿（**注意**：`layout_of` 收的是"每请求几枚草稿"，行数 = 草稿数 + 1）
drafts_per_req = [K] * 3
rows_per_req = [K + 1] * 3
cu = np.concatenate(([0], np.cumsum(rows_per_req)))
chunk_counts = {limit: len(list(_iter_request_chunks(cu, limit))) for limit in (4, 8, 11, 12)}
check("G2. 切块不切开请求：上限 4→3 块、8→2 块、11→2 块、12→1 块",
      (chunk_counts[4], chunk_counts[8], chunk_counts[11], chunk_counts[12]) == (3, 2, 2, 1),
      chunk_counts)


def verify_in_chunks_case(chunk_limit, *, stale_np_extra=0):
    """用替身采样器驱动真正的 `_verify_in_chunks`（只验切块算术）。"""
    from types import SimpleNamespace
    from minivllm.worker.gpu.spec_decode.rejection_sampler import RejectionSampler as RS

    cu_np, local_pos_np, starts = layout_of(drafts_per_req)
    num_logits = int(cu_np[-1])
    gen = torch.Generator().manual_seed(75)
    target = (torch.randn(num_logits, VOCAB, generator=gen) * 1.5).to(DEVICE)
    draft_logits = (torch.randn(MAX_NUM_REQS, K, VOCAB, generator=gen) * 2.0).to(DEVICE)
    for slot in (5, 0, 3):
        draft_logits[slot, :, slot % VOCAB] += 6.0
    draft_sampled = torch.zeros(num_logits, dtype=torch.int32, device=DEVICE)
    for r, start in enumerate(starts):
        for i in range(K):
            draft_sampled[start + i + 1] = (r + i) % VOCAB
    temp = torch.ones(MAX_NUM_REQS, dtype=torch.float32, device=DEVICE)
    temp[0], temp[5], temp[3] = 1.5, 1.0, 2.0          # 三档温度：随机行都要走概率分支
    seeds = torch.arange(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE) * 7 + 1

    class _Identity:
        logprobs_mode = "raw_logprobs"
        use_fp64_gumbel = False

        def __init__(self):
            self.sampling_states = SimpleNamespace(
                temperature=SimpleNamespace(gpu=temp),
                seeds=SimpleNamespace(gpu=seeds),
                max_num_logprobs=lambda idx_np: 2)

        def apply_sampling_params(self, logits, *a, **kw):
            return logits

    slots = [5, 0, 3]
    batch = SimpleNamespace(
        num_reqs=3,
        cu_num_logits_np=cu_np + (np.arange(4) * stale_np_extra if stale_np_extra else 0),
        cu_num_logits=torch.from_numpy(cu_np).to(DEVICE),
        idx_mapping=torch.tensor(slots, dtype=torch.int64, device=DEVICE),
        idx_mapping_np=np.array(slots, dtype=np.int64),
        expanded_idx_mapping=torch.tensor(slots, dtype=torch.int64,
                                          device=DEVICE).repeat_interleave(K + 1),
        expanded_local_pos=torch.from_numpy(local_pos_np).to(DEVICE))
    rs = RS(_Identity(), SpeculativeConfig(method="ngram", num_speculative_tokens=K), DEVICE)
    return rs._verify_in_chunks(
        target, batch, draft_logits, draft_sampled,
        torch.arange(num_logits, dtype=torch.int64, device=DEVICE) + 200, chunk_limit, 2)


single = verify_in_chunks_case(10**9)
multi = verify_in_chunks_case(K + 1)


def same_prefix(a, b):
    a_s, a_n, a_lp = a
    b_s, b_n, b_lp = b
    if not torch.equal(a_n, b_n):
        return False
    for r in range(a_n.numel()):
        n = int(a_n[r])
        if not torch.equal(a_s[r, :n], b_s[r, :n]):
            return False
    return (torch.equal(a_lp.logprob_token_ids, b_lp.logprob_token_ids)
            and torch.equal(a_lp.selected_token_ranks, b_lp.selected_token_ranks)
            and torch.allclose(a_lp.logprobs, b_lp.logprobs, atol=1e-6))


check("G3. 单块 vs 多块：sampled/num_sampled/logprobs 逐值一致（q 仍按全局 slot 取）",
      same_prefix(single, multi),
      f"num_sampled 单块={single[1].tolist()} 多块={multi[1].tolist()}、"
      f"logprobs 行={single[2].logprob_token_ids.shape}")

from minivllm.worker.gpu.spec_decode import rejection_sampler as rs_mod  # noqa: E402

calls = []
original_iter = rs_mod._iter_request_chunks


def spy(cu_num_logits, max_chunk_logits):
    for chunk in original_iter(cu_num_logits, max_chunk_logits):
        calls.append(chunk)
        yield chunk


rs_mod._iter_request_chunks = spy
try:
    stale = verify_in_chunks_case(20, stale_np_extra=2)
    stale_calls = list(calls)
    calls.clear()
    verify_in_chunks_case(K + 1)
    fresh_calls = list(calls)
finally:
    rs_mod._iter_request_chunks = original_iter
stale_cu = np.concatenate(([0], np.cumsum(rows_per_req))) + np.arange(4) * 2
check("G4. 陈旧 CPU offsets 不参与挑块边界（行数≤上限→单块）；新鲜 offsets 超上限→真的切块",
      stale_calls == [] and fresh_calls and
      list(_iter_request_chunks(stale_cu, 8)) != list(_iter_request_chunks(cu, 8))
      and same_prefix(single, stale),
      f"陈旧时切块调用={stale_calls}、新鲜时={fresh_calls}、"
      f"陈旧边界={list(_iter_request_chunks(stale_cu, 8))} vs 新鲜={list(_iter_request_chunks(cu, 8))}")

# ================================================ H. V1 regression

k = 3
batch = 3
drafts = [[1, 2, 3], [-1, 0, 4], [5, 6, 7]]
meta = make_metadata(drafts, device=DEVICE)
num_tokens = sum(len(d) for d in drafts)
v1_rows = torch.full((num_tokens + batch, 16), 1.0 / 16)
v1_rows[:, 0] = 1.0
logits = torch.log(v1_rows).to(DEVICE)
bonus = torch.zeros(batch, 1, dtype=torch.int32, device=DEVICE)
sm = sampling_metadata([1.0, 0.0, 2.0], drafts, generators={})
uniform = (torch.arange(1, num_tokens + 1, dtype=torch.float64) / (num_tokens + 1)).to(DEVICE)
recovered = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE) % 16
cond = torch.tensor(CONDITIONAL, dtype=torch.float32, device=DEVICE)
with patch_random_sources(uniform, recovered):
    mine_out = run_v1(meta, logits, None, bonus, sm, synthetic_mode=True,
                      synthetic_conditional_rates=cond)
    up_out = run_upstream_v1(meta, logits, None, bonus, [1.0, 0.0, 2.0], synthetic_mode=True,
                             synthetic_conditional_rates=cond)
check("H1. V1 synthetic 与上游逐值差分（钉死随机源：均匀数 + recovered）",
      torch.equal(mine_out, up_out), f"\n本项目\n{mine_out}\n上游\n{up_out}")

v1_batch = 6000
drafts_v1 = [[1, 2, 3]] * v1_batch
meta_v1 = make_metadata(drafts_v1, device=DEVICE)
rows_v1 = torch.full((v1_batch * (k + 1), 16), 1.0 / 16)
rows_v1[:, 0] = 1.0
logits_v1 = torch.log(rows_v1).to(DEVICE)
bonus_v1 = torch.zeros(v1_batch, 1, dtype=torch.int32, device=DEVICE)
gens = {r: torch.Generator(device=DEVICE).manual_seed(100_000 + r) for r in range(v1_batch)}
sm_v1 = sampling_metadata([1.0] * v1_batch, drafts_v1, generators=gens)
out_v1 = run_v1(meta_v1, logits_v1, None, bonus_v1, sm_v1, synthetic_mode=True,
                synthetic_conditional_rates=cond)
tau_v1 = (out_v1 != -1).sum(dim=1) - 1
dist_v1 = torch.bincount(tau_v1.long(), minlength=k + 1).double() / v1_batch
dist_v2 = tau_distribution(n_syn, K)
synthetic_ref = torch.tensor([1.0 - RATES[0], RATES[0] - RATES[1],
                              RATES[1] - RATES[2], RATES[2]], dtype=torch.float64)
check("H2. V1 与 V2 的 synthetic 语义一致（τ 分布互差 < 0.02，且都贴合配置 [0.2,0.4,0.2,0.2]）",
      total_variation(dist_v1, dist_v2) < 0.02
      and total_variation(dist_v1, synthetic_ref) < 0.02,
      f"V1 {[round(float(x), 4) for x in dist_v1]} vs V2 {[round(float(x), 4) for x in dist_v2]}"
      f" vs 配置 {[round(float(x), 4) for x in synthetic_ref]}")

# ================================================ 汇总

print()
if FAILED:
    print(f"失败 {len(FAILED)} 项：" + "、".join(FAILED))
    raise SystemExit(1)
print(f"全部通过（{PASSED} 项）")
