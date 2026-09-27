"""Triton 内核：counter RNG 与拒绝验证的批量判定/抽样（第五十六关）。

分三段，都是「固定阶段」的小内核，不追求一个巨型 kernel：

1. `event_uniform_kernel`  —— 按（请求 seed, 事件编号, 事件类型, token 下标）算 uniform。
   CPU 侧的参考实现是 `rejection_rng.event_uniform()`，两者必须**逐位一致**（有对照测试）。
2. `accept_prefix_kernel` —— 每个请求一段：沿 K 个位置推进「接受前缀」，遇到首拒绝或
   接受终止 token 就停；同时数出**实际消费**了几个接受事件。
3. `categorical_kernel`   —— 纠正/bonus 抽样：在 151936 的词表上分块做 Gumbel-max 归约，
   不给小词表硬编码超大单程序。

约定（与需求 §4 一致）：只有**真正发生**的事件才推进计数器。kernel 里可以为整段 K
提前算好候选随机数（counter-based 天然允许），只有实际用到的那些体现在增量里。
"""

import torch
import triton
import triton.language as tl

# 与 rejection_rng.py 共用同一套常数与事件编号
from .rejection_rng import ACCEPT, CATEGORICAL, PHILOX_ROUNDS, UNIFORM_SCALE, UNIFORM_SHIFT

BLOCK_V = 4096          # 词表分块大小（151936 词表 -> 38 块）
SENTINEL = -1           # 未使用位置的哨兵

# kernel 里要用的模块级常量必须是 tl.constexpr 实例（Triton 的硬要求）
_SHIFT = tl.constexpr(UNIFORM_SHIFT)
_SCALE = tl.constexpr(UNIFORM_SCALE)
_ACCEPT = tl.constexpr(ACCEPT)
_CATEGORICAL = tl.constexpr(CATEGORICAL)


@triton.jit
def _uniform_from(out_word):
    """uint32 -> [0,1) 的 FP32：取高 24 位当尾数。与 CPU 参考逐位一致。"""
    word = tl.to_tensor(out_word)
    return (word >> _SHIFT).to(tl.float32) * _SCALE


@triton.jit
def _event_uniform(seed_lo, seed_hi, event_index, event_type, token_index,
                   ROUNDS: tl.constexpr):
    """一个逻辑事件的 uniform：Philox-4x32-10，计数器 = (事件编号, 类型, token, 0)。

    入口先把三个计数器字 `tl.to_tensor()`：调用点常常直接传 Python 字面量
    （比如接受事件的 token 下标恒为 0），不转成张量的话 Triton 会在编译期炸在 `.to()` 上。
    """
    seed = (tl.to_tensor(seed_hi).to(tl.uint64) << 32) | tl.to_tensor(seed_lo).to(tl.uint64)
    c0 = tl.to_tensor(event_index).to(tl.uint32)
    c1 = tl.to_tensor(event_type).to(tl.uint32)
    c2 = tl.to_tensor(token_index).to(tl.uint32)
    c3 = tl.zeros_like(c0)
    out0, _, _, _ = tl.philox(seed, c0, c1, c2, c3, n_rounds=ROUNDS)
    return _uniform_from(tl.to_tensor(out0))


@triton.jit
def event_uniform_kernel(seed_lo_ptr, seed_hi_ptr, index_ptr, type_ptr, token_ptr,
                         out_ptr, N, ROUNDS: tl.constexpr, BLOCK: tl.constexpr):
    """把 N 个事件的 uniform 一次算出来（测试与 kernel 内部共用同一段逻辑）。"""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    seed_lo = tl.load(seed_lo_ptr + offsets, mask=mask, other=0).to(tl.int64)
    seed_hi = tl.load(seed_hi_ptr + offsets, mask=mask, other=0).to(tl.int64)
    index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    event_type = tl.load(type_ptr + offsets, mask=mask, other=0).to(tl.int64)
    token = tl.load(token_ptr + offsets, mask=mask, other=0).to(tl.int64)
    value = _event_uniform(seed_lo, seed_hi, index, event_type, token, ROUNDS)
    tl.store(out_ptr + offsets, value, mask=mask)


def event_uniforms(seed, event_index, event_type, token_index):
    """便捷入口：给一组 CPU 侧的事件参数，返回 GPU 上算出来的 uniform（对照用）。"""
    seed_lo = torch.tensor([(int(seed) & 0xFFFFFFFF)] * len(event_index),
                           dtype=torch.int64, device="cuda")
    seed_hi = torch.tensor([(int(seed) >> 32)] * len(event_index),
                           dtype=torch.int64, device="cuda")
    index = torch.tensor(event_index, dtype=torch.int64, device="cuda")
    types = torch.tensor(event_type, dtype=torch.int64, device="cuda")
    tokens = torch.tensor(token_index, dtype=torch.int64, device="cuda")
    out = torch.empty(len(event_index), dtype=torch.float32, device="cuda")
    n = len(event_index)
    event_uniform_kernel[(triton.cdiv(n, 256),)](seed_lo, seed_hi, index, types, tokens, out,
                                                n, ROUNDS=PHILOX_ROUNDS, BLOCK=256)
    return out


@triton.jit
def verify_prefix_kernel(ratio_ptr, eos_ptr, invalid_ptr, pos_start_ptr, k_ptr,
                         seed_lo_ptr, seed_hi_ptr, counter_ptr,
                         accepted_ptr, kind_ptr, consumed_ptr, error_ptr,
                         KMAX: tl.constexpr, ROUNDS: tl.constexpr):
    """一个 program 一条请求：沿 K 个位置推进接受前缀（需求 §3 的规则）。

    `ratio` 是预先在 GPU 上 gather 好的 `p[d]/q[d]`（ngram 时就是 `p[d]`）。
    `kind`：0 = 全接受、1 = 首拒绝、2 = 接受的草稿是终止 token。
    `consumed` 只数**真的抽了 uniform** 的那些位置（0 < ratio < 1），
    必接受 / 必拒绝都不消费——这样计数器的增量与 CPU 参考路径逐条对得上。
    """
    b = tl.program_id(0)
    start = tl.load(pos_start_ptr + b)
    k = tl.load(k_ptr + b)
    seed_lo = tl.load(seed_lo_ptr + b).to(tl.int64)
    seed_hi = tl.load(seed_hi_ptr + b).to(tl.int64)
    base = tl.load(counter_ptr + b).to(tl.int64)

    accepted = 0
    consumed = 0
    kind = 0
    error = 0
    for i in tl.static_range(KMAX):
        if (i < k) and (kind == 0) and (error == 0):
            pos = start + i
            ratio = tl.load(ratio_ptr + pos)
            if tl.load(invalid_ptr + pos) != 0:
                error = 1                      # 非法提议（例如 q[d] = 0）
            elif ratio >= 1.0:
                accepted += 1                  # 必接受：不消费随机事件
                if tl.load(eos_ptr + pos) != 0:
                    kind = 2
            elif ratio <= 0.0:
                kind = 1                       # 必拒绝：同样不消费
            else:
                u = _event_uniform(seed_lo, seed_hi, base + consumed, _ACCEPT, 0, ROUNDS)
                consumed += 1
                if u < ratio:
                    accepted += 1
                    if tl.load(eos_ptr + pos) != 0:
                        kind = 2
                else:
                    kind = 1
    tl.store(accepted_ptr + b, accepted)
    tl.store(kind_ptr + b, kind)
    tl.store(consumed_ptr + b, consumed)
    tl.store(error_ptr + b, error)


@triton.jit
def sample_token_kernel(weight_ptr, seed_lo_ptr, seed_hi_ptr, event_index_ptr,
                        out_token_ptr, out_error_ptr, VOCAB,
                        BLOCK_V: tl.constexpr, ROUNDS: tl.constexpr):
    """一次 categorical 抽样：**指数竞赛** + 词表分块归约。

    对每个 token 取 `e_i = -log(u_i) / w_i`（`u_i` 是这个事件在 token i 上的随机数），
    然后取最小的那个——独立指数分布竞速恰好等价于按 `w` 抽样。两个要点：

    - `w` 不必归一化：全体乘一个正常数不改变 argmin，所以纠正分布**不用**在这里归一化；
    - `w_i = 0` 的 token 给 `+inf`，永远不会被选中（`max(p-q,0)` 的零质量就是这样排除的）；
    - `u = (x + 0.5) · 2**-32` 在 FP64 里算：保证落在 (0,1) 开区间内（不会出现
      `log(0)` 或 `log(1)` 那种边界），偏差是 2**-32 量级；
    - 并列时取**较小下标**，与 CPU oracle 的 `argmin` 行为对齐。
    """
    d = tl.program_id(0)
    seed_lo = tl.load(seed_lo_ptr + d).to(tl.int64)
    seed_hi = tl.load(seed_hi_ptr + d).to(tl.int64)
    event_index = tl.load(event_index_ptr + d).to(tl.int64)

    best_e = float("inf")
    best_i = 0
    for tile in range(0, tl.cdiv(VOCAB, BLOCK_V)):
        offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offs < VOCAB
        weight = tl.load(weight_ptr + d * VOCAB + offs, mask=mask, other=0.0).to(tl.float64)
        word = _event_uniform(seed_lo, seed_hi, event_index, _CATEGORICAL, offs, ROUNDS)
        u = (word.to(tl.uint32).to(tl.float64) + 0.5) * (1.0 / 4294967296.0)
        race = tl.where(weight > 0.0, -tl.math.log(u) / weight, float("inf"))
        race = tl.where(mask, race, float("inf"))
        tile_min = tl.min(race, axis=0)
        tile_index = tl.min(tl.where(race == tile_min, offs, VOCAB), axis=0)
        better = (tile_min < best_e) or ((tile_min == best_e) and (tile_index < best_i))
        best_e = tl.where(better, tile_min, best_e)
        best_i = tl.where(better, tile_index, best_i)

    tl.store(out_token_ptr + d, best_i)
    tl.store(out_error_ptr + d, tl.where(best_e == float("inf"), 1, 0))
