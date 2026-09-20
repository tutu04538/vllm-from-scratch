"""Triton 采样后端：**最终那一步选 token** 真的在 Triton 里做。

前面那些处理（惩罚、温度、top-k、softmax、top-p）仍留在 Torch —— 这一关允许，
排序和累计概率不值得为它写 kernel。这里要保证的是：

    greedy 的最大值/索引归约      -> Triton
    random 的按分布选择            -> Triton（Gumbel-max）

不能只在 Triton 里缩放 logits，最后仍调用 Torch 的 argmax / multinomial。

**不给全词表开一个巨大的 tl.arange**：真实词表 151936，按 BLOCK 分块，
每块先算出局部最优，再来一个第二阶段把几十个局部结果合并。

Gumbel-max：对最终保留的 logit z_i 取 argmax_i [z_i - log(-log(u_i))]，u_i ~ U(0,1)。
精确运算下与「按 softmax(z) 抽样」等价。被过滤掉的 token 仍是 -inf，
减去一个有限噪声还是 -inf，不会重新进入候选。
"""

import torch

import triton
import triton.language as tl

# 一次归约覆盖的词表分块大小。1024 让真实词表分成 149 个 program，
# 够铺满 82 个 SM；4096 只有 38 个 program，反而喂不饱机器。
BLOCK = 1024
# 每个 program 用几个 warp。实测 4 最好（1 个太少、2 个更差、8 个略降）
NUM_WARPS = 4
# 第二阶段最多合并多少个局部结果（PARTS 取 2 的幂，词表 / BLOCK 不能超过它）
MAX_PARTS = 256


@triton.jit
def _argmax_partial_kernel(x_ptr, val_ptr, idx_ptr, n,
                           N: tl.constexpr, BLOCK: tl.constexpr):
    # 网格是 (行, 块)：整批请求一次发射，发射次数与请求数无关
    row = tl.program_id(0)
    part = tl.program_id(1)
    offs = part * BLOCK + tl.arange(0, BLOCK)
    x_ptr = x_ptr + row * N
    val_ptr = val_ptr + row * tl.num_programs(1)
    idx_ptr = idx_ptr + row * tl.num_programs(1)
    pid = part
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
    best = tl.max(x, axis=0)
    # 并列取最小 token id：取所有等于最大值的下标里最小的那个。
    # 被 mask 掉的位置（以及整块都在词表外的）填 n 当「无效」，不会赢过真实下标
    idx = tl.min(tl.where(mask & (x == best), offs, n), axis=0)
    tl.store(val_ptr + pid, best)
    tl.store(idx_ptr + pid, idx)


@triton.jit
def _argmax_merge_kernel(val_ptr, idx_ptr, out_ptr, nparts,
                         N: tl.constexpr, PARTS: tl.constexpr):
    row = tl.program_id(0)
    val_ptr = val_ptr + row * nparts
    idx_ptr = idx_ptr + row * nparts
    offs = tl.arange(0, PARTS)
    mask = offs < nparts
    val = tl.load(val_ptr + offs, mask=mask, other=float("-inf"))
    idx = tl.load(idx_ptr + offs, mask=mask, other=N)
    best = tl.max(val, axis=0)
    # 每块的 idx 已经是块内最小下标，且块按 pid 递增 —— 再取一次最小就是全局最小
    out = tl.min(tl.where(mask & (val == best), idx, N), axis=0)
    # 词表非空时一定有真实候选；这里再夹一下，避免极端情况返回 N
    out = tl.minimum(out, N - 1)
    tl.store(out_ptr + row, out)


@triton.jit
def _gumbel_partial_kernel(z_ptr, val_ptr, idx_ptr, n, seed_ptr, off_ptr,
                           N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    # 种子和随机数位置都是**按请求**取的：同一请求换一行也不换流
    seed = tl.load(seed_ptr + row)
    offset_base = tl.load(off_ptr + row)
    offs = part * BLOCK + tl.arange(0, BLOCK)
    z_ptr = z_ptr + row * N
    val_ptr = val_ptr + row * tl.num_programs(1)
    idx_ptr = idx_ptr + row * tl.num_programs(1)
    pid = part
    mask = offs < n
    z = tl.load(z_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
    # 随机数的「位置」按全局下标走：同一个请求这一轮用 [base, base+n)，
    # 下一轮 base += n —— 不换流、也不从头开始
    u = tl.rand(seed, offset_base + offs)
    u = tl.minimum(tl.maximum(u, 1e-7), 1.0 - 1e-7)
    g = z - tl.log(-tl.log(u))
    # 被过滤的 token 是 -inf，减去有限噪声还是 -inf，不会因为加噪声复活
    g = tl.where(mask, g, float("-inf"))
    best = tl.max(g, axis=0)
    idx = tl.min(tl.where(mask & (g == best), offs, n), axis=0)
    tl.store(val_ptr + pid, best)
    tl.store(idx_ptr + pid, idx)


def _pick_batch(z, seeds=None, offsets=None):
    """z 是 [M, V] 的 FP32 logits（过滤过的位置是 -inf）。返回 [M] 的 token id。

    两次 kernel 发射覆盖整批：第一次按 (行, 块) 求局部最优，第二次每行合并几十个局部结果。
    发射次数与请求数无关——逐请求发射会在请求多的时候被发射开销吃掉收益。
    """
    m, n = z.shape
    nparts = triton.cdiv(n, BLOCK)
    if nparts > MAX_PARTS:
        raise ValueError(f"词表 {n} 需要 {nparts} 个分块，超过 {MAX_PARTS}；请调大 BLOCK")
    if z.dtype != torch.float32 or not z.is_contiguous():
        z = z.to(torch.float32).contiguous()
    vals = torch.empty(m, nparts, device=z.device, dtype=torch.float32)
    idxs = torch.empty(m, nparts, device=z.device, dtype=torch.int32)
    out = torch.empty(m, device=z.device, dtype=torch.int64)
    parts_pow2 = triton.next_power_of_2(nparts)
    if seeds is None:
        _argmax_partial_kernel[(m, nparts)](z, vals, idxs, n, N=n, BLOCK=BLOCK,
                                            num_warps=NUM_WARPS)
    else:
        _gumbel_partial_kernel[(m, nparts)](z, vals, idxs, n, seeds, offsets, N=n, BLOCK=BLOCK,
                                            num_warps=NUM_WARPS)
    _argmax_merge_kernel[(m,)](vals, idxs, out, nparts, N=n, PARTS=parts_pow2,
                               num_warps=NUM_WARPS)
    return out


class TritonSampler:
    """最终选择在 Triton kernel 里完成；过滤与惩罚沿用 Torch（本关允许）。"""

    name = "triton"

    def _filtered_logits(self, row, params, state):
        """Torch 侧把 logits 处理成「候选为真实分数、其余为 -inf」的一维 FP32。"""
        from .sampling import _filter_and_probs
        if params.is_greedy:
            return row
        probs = _filter_and_probs(row, params)
        return torch.where(probs > 0, torch.log(probs), float("-inf"))

    def select(self, row, params, state):
        """单行便利入口，和 TorchSampler 对称；内部仍走批量路径。"""
        return self.select_batch([row], [params], [state])[0]

    def select_batch(self, rows, params_list, states):
        """rows 是每个请求的 FP32 工作副本；返回与之一一对应的 token id 列表（GPU 标量）。

        贪心的和随机的各凑一批：每批两次发射，总共最多四次，与请求数无关。
        """
        out = [None] * len(rows)
        greedy = [i for i, p in enumerate(params_list) if p.is_greedy]
        if greedy:
            ids = _pick_batch(torch.stack([rows[i] for i in greedy]))
            for k, i in enumerate(greedy):
                out[i] = ids[k]
        random_rows = [i for i, p in enumerate(params_list) if not p.is_greedy]
        if random_rows:
            device = rows[0].device
            z = torch.stack([self._filtered_logits(rows[i], params_list[i], states[i])
                             for i in random_rows])
            seeds = torch.tensor([states[i].rng_seed for i in random_rows],
                                 dtype=torch.int32, device=device)
            # offset 是「这个请求累计消耗了多少个随机数」= 抽样次数 × 词表大小。
            # 151936 的整词表只要一万多次抽样就越过 int32 上限，所以用 int64：
            # Triton 的 randint4x 会把 >32 位的部分取高位一起送进 Philox。
            # 取模归零复用旧流才是错的。
            offsets = torch.tensor([states[i].rng_offset for i in random_rows],
                                   dtype=torch.int64, device=device)
            ids = _pick_batch(z, seeds, offsets)
            for k, i in enumerate(random_rows):
                states[i].rng_offset += z.shape[1]   # 这个请求消耗了 n 个随机数
                out[i] = ids[k]
        return out
