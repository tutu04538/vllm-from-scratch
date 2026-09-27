"""拒绝验证的请求级随机流：counter-based RNG（第五十六关，triton 后端专用）。

需求 §4 把规则定成：

    随机数 = f(请求 seed, 逻辑事件编号, 事件类型, 词表 token ID)

`f` 用 **Philox-4x32-10**（Random123 的标准 counter-based 生成器，常数列在下面）。
选它而不是「每条请求预抽 K 个 uniform」的原因很直接：预抽之后遇到首拒绝或 EOS，
**后面根本没有发生的随机事件**也会算进去，于是下一轮的流位置就错了。counter-based
的好处是「事件的随机数只由它的编号决定」——没发生的事件编号根本不用，自然不会推进。

计数器的四个 32 位字（位宽写清楚，避免静默截断）：

    c0 = 事件编号 event_index（int64 截到 32 位；每请求独立计数，见下）
    c1 = 事件类型 event_type（ACCEPT=0 / CATEGORICAL=1）
    c2 = 词表内下标 token_index（接受事件恒为 0；categorical 事件按 token ID 区分）
    c3 = 0（保留）
    k0 = seed 低 32 位，k1 = seed 高 32 位

**消费规则**（与 CPU 参考路径的 `verify_drafts_random()` 一致，测试会逐条数）：

- `0 < 接受概率 < 1`：消费一个 ACCEPT 事件；
- 必接受（概率 ≥ 1）/ 必拒绝（概率 ≤ 0）：不消费；
- 纠正或 bonus 抽样：消费一个 CATEGORICAL 事件（词表内部随机数由 token ID 区分）；
- greedy：不消费任何事件；接受 EOS 之后也不再消费。

`event_index` 是本请求**已消费事件数**（`SequenceConfig.rejection_rng_counter`）加本轮
第几个候选事件：所以同一段逻辑事件序列总是拿到同一组随机数（请求重排、别的请求插入、
抢占恢复都不影响），而没发生的事件不会推进计数器。

GPU 侧允许**提前算好候选随机数**（kernel 里对 K 个位置一次算完），只有实际发生的
那些会体现在计数器的增量里——这正是 counter-based 的意义。
"""

# Philox-4x32-10 的常数（Random123 / 本机 triton.language.random 用的是同一套）
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
PHILOX_ROUNDS = 10
MASK32 = 0xFFFFFFFF

# 事件类型
ACCEPT = 0
CATEGORICAL = 1

# uint32 -> [0,1) 的换算：取高 24 位当尾数，逐位精确、CPU/GPU 完全一致
UNIFORM_SHIFT = 8
UNIFORM_SCALE = 1.0 / (1 << 24)


def _mulhilo(a, b):
    """无符号 32x32 -> 64 位乘积，返回 (低 32 位, 高 32 位)。"""
    product = (a & MASK32) * (b & MASK32)
    return product & MASK32, (product >> 32) & MASK32


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Philox-4x32-10 的一轮输出（4 个 uint32）。CPU 参考实现，与 kernel 逐位一致。"""
    for round_index in range(PHILOX_ROUNDS):
        lo0, hi0 = _mulhilo(PHILOX_M0, c0)
        lo1, hi1 = _mulhilo(PHILOX_M1, c2)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & MASK32, lo1, (hi0 ^ c3 ^ k1) & MASK32, lo0
        if round_index != PHILOX_ROUNDS - 1:
            k0 = (k0 + PHILOX_W0) & MASK32
            k1 = (k1 + PHILOX_W1) & MASK32
    return c0, c1, c2, c3


def event_words(seed, event_index, event_type, token_index):
    """把一个逻辑事件映射成 Philox 的（计数器, 密钥）两组 32 位字。"""
    if not 0 <= event_index < (1 << 32):
        raise ValueError(f"事件编号必须落在 [0, 2**32) 内，收到 {event_index}")
    if not 0 <= event_type <= MASK32 or not 0 <= token_index <= MASK32:
        raise ValueError(f"事件类型/token 下标必须落在 [0, 2**32) 内，收到 "
                         f"{event_type}/{token_index}")
    seed &= (1 << 64) - 1
    return ((event_index, event_type, token_index, 0),
            (seed & MASK32, (seed >> 32) & MASK32))


def event_word(seed, event_index, event_type, token_index):
    """一个逻辑事件对应的**原始 32 位随机字**（Philox 输出的第一个字）。

    categorical 抽样要的是这个原始字（kernel 里 `(x + 0.5)·2**-32` 那种换算），
    接受判断要的是 `event_uniform()` 那个 float。两个入口共用同一次 Philox 计算。
    """
    counter, key = event_words(seed, event_index, event_type, token_index)
    return philox4x32_10(*counter, *key)[0]


def event_uniform(seed, event_index, event_type, token_index):
    """一个逻辑事件对应的 uniform（[0,1)，FP32 可精确表示）。

    取原始字的高 24 位：这样 CPU 的 Python 参考与 Triton kernel 得到**同一个** float，
    不会因为换算方式不同而分叉（有逐位对照测试）。
    """
    return (event_word(seed, event_index, event_type, token_index) >> UNIFORM_SHIFT) * UNIFORM_SCALE


def exponential_race(seed, event_index, weights):
    """CPU 侧的 categorical 抽样：与 kernel 同一套算法（指数竞赛 + 并列取小下标）。

    `e_i = -log(u_i) / w_i`，`u_i = (word_i + 0.5) · 2**-32` 落在 (0,1) 开区间内；
    取最小者。`w_i <= 0` 给 +inf，永不入选。
    """
    import math
    best_value, best_index = math.inf, 0
    for index, weight in enumerate(weights):
        if weight <= 0:
            continue
        word = event_word(seed, event_index, CATEGORICAL, index)
        u = (word + 0.5) / 4294967296.0
        value = -math.log(u) / weight
        if value < best_value:
            best_value, best_index = value, index
    return best_index, math.isinf(best_value)


def derive_rejection_seed(seed):
    """由请求的 `seed` 稳定派生拒绝验证的种子（与 draft 侧同样的规则：线性同余混合）。

    `seed=None` 时返回 None，由 `make_rejection_seed()` 从全局随机源取一个
    （与 target/draft 两侧一致：那时本来就不承诺复现）。
    """
    from .draft import DRAFT_SEED_INCREMENT, DRAFT_SEED_MULTIPLIER, DRAFT_SEED_MODULUS
    if seed is None:
        return None
    # 换一组常数：同一个请求的 draft 流与拒绝验证流不能是同一个种子
    return (seed * DRAFT_SEED_MULTIPLIER + DRAFT_SEED_INCREMENT * 3 + 1) % DRAFT_SEED_MODULUS


def make_rejection_seed(params):
    """请求级的拒绝验证种子：有 `seed` 就派生，没有就从**全局随机源**取一个。

    与 draft 侧 `draft.make_draft_generator()` 同一套规则：`seed=None`（调用方没要求
    复现）时用不传 `generator=` 的 `torch.randint`，也就是 PyTorch 的默认生成器；
    **本包从不播种它**，只在请求创建时碰一次，绝不在 `step()` 里抽——否则别的请求
    的推进会扰动已有请求的随机流。上界取 `2**63 - 1`（`2**63` 超出 int64，会抛
    「Overflow when unpacking long long」）。

    **为什么不干脆退化成 0**：那样所有未播种的请求会共用同一条随机流——同一个事件
    编号拿到同一个随机数，它们的接受/拒绝会完全相关。那比「不可复现」坏得多。

    贪心请求不抽随机数（接受判断是纯比较），照样给个种子：后端只认字段，不在
    「要不要建这个字段」上分叉；它也确实用不到。
    """
    from .draft import DRAFT_SEED_MODULUS
    import torch
    seed = derive_rejection_seed(params.seed)
    if seed is None:
        seed = int(torch.randint(0, DRAFT_SEED_MODULUS - 1, (1,)).item())
    return seed
