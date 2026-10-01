"""top-k / top-p 筛选与随机抽样（对应 vLLM `v1/sample/ops/topk_topp_sampler.py` 的 Torch 路径）。

两件事：

    apply_top_k_top_p(logits, k, p)   把不候选的 logits 打成 -inf（**原地**）
    random_sample(probs, generators)  按概率抽一个 token

### top-p 的两个细节

`apply_top_k_top_p_pytorch` 是 vLLM 的参考实现，做法是**升序排序**后在排序空间里做掩码：

    probs_sum = cumsum(softmax(sorted_logits)[::-1 的补集...])
    top_p_mask = probs_sum <= 1 - p       # 从"小的那头"累加
    top_p_mask[:, -1] = False             # 最大的那个永远保留 → **至少一个候选**

这解决了两个边界：① 累积和刚好跨过阈值的那个 token（"边界 token"）要**保留**——用
`<= 1 - p` 而不是 `<`，且掩码作用在"小的那头"，所以边界 token 落在保留侧；② 概率分布极端
（p 很小或只有一个非零）时也一定有一个候选，不会全 -inf 导致 softmax 出 NaN。

### 抽样：指数竞赛（不是 `torch.multinomial`）

    q_i ~ Exp(1)，取 argmax(probs_i / q_i)

这个 argmax 恰好以 `probs` 为分布（Gumbel-max 的等价形式），而且**整批一次算完**、
不需要把 probs 拉回 CPU——`torch.multinomial` 会引入一次同步，vLLM 的注释就是为这个才自己写。

**随机源**：每个请求自己的 generator（`generators[row]`）只作用在**它那一行**的噪声上；
没有 seed 的行用全局 RNG。第 56 关的拒绝采样用的也是这套"指数竞赛"，只是那里比较的是
`-log(u)/w`——两者是同一件事的两种写法。
"""

import torch

# 与 vLLM 一致：温度低于它就当贪心（见 metadata.SAMPLING_EPS 的说明）
SAMPLING_EPS = 1e-5


def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None,
                      p: torch.Tensor | None) -> torch.Tensor:
    """原地把非候选的 logits 打成 -inf。`k`/`p` 为 None 表示该维度不筛。

    **调用方保证 `0 < k < vocab_size`**（与 vLLM 同款约定）：`k >= V` 会让 `V - k` 变负数、
    `k <= 0` 会让它变成 V，`gather` 两种都会当场报越界。这不是靠算子自己兜底，而是在
    **批层面**就归类掉——不需要筛的行在张量里写的是 `vocab_size`（见 `InputBatch`），
    那样 `V - k = 0`，阈值取到最小值，等于什么都不屏蔽。
    """
    if k is None and p is None:
        return logits
    # 升序排序：从小到大扫，方便"从概率小的一头开始切"
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # 第 k 大的那个值：升序排序后，top-k 是**最后 k 个**，其中最小的是下标 `V - k`。
        # 先算出这个**位置**，再用 gather 取出该位置上的**值**（阈值）。
        # 注意 `top_k_mask` 这个名字在这两行里换了两次身份：位置 → 阈值 → 掩码（vLLM 也这么写）
        top_k_index = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_index.unsqueeze(dim=1))
        # 严格小于阈值的丢掉；**等于阈值的留下**，所以并列时可能比 k 多几个
        logits_sort.masked_fill_(logits_sort < top_k_mask, -float("inf"))

    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # 累积和还没到 1-p 的那些（概率最小的一批）全部丢掉
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False           # 至少留一个（最大的那个）
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def random_sample(probs: torch.Tensor, generators: dict[int, torch.Generator]) -> torch.Tensor:
    """指数竞赛抽样。`generators` 是"紧凑行号 → 该行的 generator"（没 seed 的行不在里面）。"""
    noise = torch.empty(probs.shape, dtype=probs.dtype, device=probs.device)
    if len(generators) != probs.shape[0]:
        # 有行用全局 RNG：整块先抽一次（覆盖所有行），下面再逐行覆盖有 generator 的那些
        noise.exponential_()
    for row, generator in generators.items():
        noise[row].exponential_(generator=generator)
    return (probs / noise).argmax(dim=-1).view(-1)


class TopKTopPSampler:
    """把"温度缩放后的 logits → 一个 token"打包成一次调用（vLLM 里它是个 nn.Module）。"""

    def __call__(self, logits: torch.Tensor, generators: dict[int, torch.Generator],
                 k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
        logits = apply_top_k_top_p(logits, k, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators)
