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

这个 argmax 恰好以 `probs` 为分布（Gumbel-max 的等价形式），而且**整批一次算完**。

vLLM 的原注释说，自己写 `random_sample` 是为了避开 `torch.multinomial` 的同步：

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.

本机 torch 2.13 **没有复现出同步**（GPU 忙时的 CPU 侧耗时并不变大），但开销差距是真的：
每次调用 CPU 侧 167 μs vs 51 μs、GPU 侧 204 μs vs 76 μs（`[8, 151936]` 的批）。多出来的部分
来自它额外挂的校验步骤——profile 里能看到 `aminmax` + `sum` + 两次 `_assert_async`
（设备端断言的机制，历史上就是靠宿主等待来检查的）。所以"为了避开同步"这句注释在当前版本
更准确的读法是：**它每一步都做了指数竞赛不需要的检查，而且那些检查在别的版本上会等 GPU**。

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
    """把"温度缩放后的 logits → 一个 token"打包成一次调用（vLLM 里它是个 nn.Module）。

    **返回值是二元组**（68 关改；与上游 `forward_native` 同签名）：`(采样结果, 要留的 logits)`。
    第二个元素只在这两种模式下非 None（`processed_logits` / `processed_logprobs`）——
    它交付的是**筛选之后**的那份 logits（top-k/top-p 已经把落选者打成 -inf），
    所以 `processed_*` 模式的 logprobs 名次与真正的采样分布严格一致（068 §3.5）。
    """

    def __init__(self, logprobs_mode: str = "raw_logprobs") -> None:
        self.logprobs_mode = logprobs_mode

    def __call__(self, logits: torch.Tensor, generators: dict[int, torch.Generator],
                 k: torch.Tensor | None, p: torch.Tensor | None
                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = apply_top_k_top_p(logits, k, p)      # 原地：落选者变 -inf
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return
