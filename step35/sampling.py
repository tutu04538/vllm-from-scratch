"""采样：把模型输出的一行 logits 变成一个 token id。

模型只负责算 logits；这里负责按**这个请求自己的**参数把它变成一个 token。
三条路径：

    greedy：只施加惩罚，然后 argmax（不算 softmax、不缩放温度、不筛选）
    random：惩罚 → 除以 temperature → top-k → softmax → top-p → 抽样
    beam  ：走独立的 beam.py，用惩罚后 logits 的完整 log_softmax 评分，不用这里的单 token 抽样

执行顺序是固定的（见 docs），顺序本身会影响结果，不能各写各的。
"""

import math

import torch


def _reject_bool(value, name):
    # bool 是 int 的子类，不特判的话 True 会被当成 1 悄悄放过
    if isinstance(value, bool):
        raise ValueError(f"{name} 不接受布尔值，收到 {value!r}")


def _check_number(value, name, low, high, low_open=False):
    _reject_bool(value, name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值，收到 {value!r}")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} 不能是 NaN 或 Inf，收到 {value!r}")
    if value < low or value > high or (low_open and value == low):
        bracket = f"({low}, {high}]" if low_open else f"[{low}, {high}]"
        raise ValueError(f"{name} 必须在 {bracket} 内，收到 {value}")
    return value


class SamplingParams:
    """一个请求的采样参数。默认值 = 贪心，也就是旧请求的行为。"""

    def __init__(self, temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                 presence_penalty=0.0, frequency_penalty=0.0, seed=None, vocab_size=None):
        self.temperature = _check_number(temperature, "temperature", 0.0, 10.0)
        if self.temperature != 0.0 and self.temperature < 0.01:
            raise ValueError(f"temperature 只能是 0（贪心）或 [0.01, 10] 内的值，收到 {self.temperature}")

        _reject_bool(top_k, "top_k")
        if not isinstance(top_k, int):
            raise ValueError(f"top_k 必须是整数，收到 {top_k!r}")
        if top_k < 0 or (vocab_size is not None and top_k > vocab_size):
            raise ValueError(f"top_k 必须在 [0, {vocab_size if vocab_size is not None else 'vocab_size'}] 内，收到 {top_k}")
        self.top_k = top_k

        # 下界是开区间：0 < top_p <= 1。top_p=0 会让保留集合空掉（除出 0/0），
        # 但任意小的正数都合法——它仍然至少保留概率最大的那一个
        self.top_p = _check_number(top_p, "top_p", 0.0, 1.0, low_open=True)
        self.repetition_penalty = _check_number(repetition_penalty, "repetition_penalty", 0.01, 100.0)
        self.presence_penalty = _check_number(presence_penalty, "presence_penalty", -2.0, 2.0)
        self.frequency_penalty = _check_number(frequency_penalty, "frequency_penalty", -2.0, 2.0)

        _reject_bool(seed, "seed")
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError(f"seed 必须是整数或 None，收到 {seed!r}")
            if not 0 <= seed <= 2 ** 63 - 1:
                raise ValueError(f"seed 必须在 [0, 2**63-1] 内，收到 {seed}")
        self.seed = seed

    @property
    def is_greedy(self):
        return self.temperature == 0.0

    @property
    def has_penalty(self):
        return (self.repetition_penalty != 1.0 or self.presence_penalty != 0.0
                or self.frequency_penalty != 0.0)

    def __repr__(self):
        return (f"SamplingParams(temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}, "
                f"repetition_penalty={self.repetition_penalty}, presence_penalty={self.presence_penalty}, "
                f"frequency_penalty={self.frequency_penalty}, seed={self.seed})")


class SamplingState:
    """一个请求的采样状态：惩罚计数 + 自己的随机数发生器。

    **按请求隔离**，不按 batch 行号。同一个请求这一轮在第 0 行、下一轮在第 2 行，
    随机数流不能因此换一条；插入别的请求也不能扰动它。
    """

    def __init__(self, params: SamplingParams, prompt_ids, device):
        self.params = params
        # repetition penalty 看的是 prompt + 已生成：prompt 那部分一开始就定下来
        self.prompt_token_ids = tuple(sorted(set(prompt_ids)))
        # presence / frequency penalty 只看**已生成**部分
        self.generated_counts = {}
        self.generated_total = 0
        self.generator = None
        # Triton 后端的随机状态：内核种子 + 这个请求已消耗的随机数个数。
        # 位置绑在**请求**上而不是 batch 行号上：同一请求这一轮在第 0 行、
        # 下一轮在第 2 行，随机数流不能因此换一条。
        self.rng_seed = 0
        self.rng_offset = 0
        if not params.is_greedy:
            self.generator = torch.Generator(device=device)
            if params.seed is not None:
                self.generator.manual_seed(params.seed)
            else:
                # **新建 Generator 不等于给了随机种子**：所有新 Generator 的初始种子
                # 都是同一个常量，不手动播种的话两个没写 seed 的请求会读同一条流。
                # 这里从全局随机源取一个，只在请求创建时播一次，不每步重置。
                self.generator.manual_seed(int(torch.randint(0, 2 ** 63 - 1, (1,)).item()))
            # 内核只吃 32 位种子，把 63 位折进去（仍然是种子的确定函数）
            seed64 = params.seed if params.seed is not None else self.generator.initial_seed()
            self.rng_seed = (seed64 ^ (seed64 >> 32)) & 0x7FFFFFFF

    def note_output_token(self, token_id):
        # 只有真实提交的输出 token 才计数：M=0、预填充中间块都不经过这里
        self.generated_counts[token_id] = self.generated_counts.get(token_id, 0) + 1
        self.generated_total += 1


def apply_penalties(row, params: SamplingParams, state: SamplingState):
    """在 FP32 工作副本上施加三种惩罚，返回新张量。

    **不原地改输入**：模型给的 logits 可能是 Graph 复用的缓冲，
    而且 FP32 张量的 `.float()` 并不产生副本。调用方已经传进来一份副本，
    这里仍然只用 out-of-place 的操作。
    """
    if not params.has_penalty:
        return row

    device = row.device

    # 顺序固定：repetition 先、frequency + presence 后。
    # 换顺序结果就不一样——需求给的例子 logit=4、r=2、p=0.5、f=0.2、count=3
    # 算的是 4/2 - 0.5 - 0.2*3 = 0.9；先减再加权除法会得到 1.45。
    if params.repetition_penalty != 1.0:
        seen = state.prompt_token_ids
        if state.generated_counts:
            seen = tuple(sorted(set(seen) | set(state.generated_counts)))
        if seen:
            ids = torch.tensor(seen, device=device, dtype=torch.long)
            r = params.repetition_penalty
            old = row.index_select(0, ids)
            # 正 logit 除以 r、负 logit 乘以 r、零不变：方向不能反，
            # 否则 -4 变成 -2，负分反而更接近零，惩罚变成了奖励
            new = torch.where(old > 0, old / r, old * r)
            row = row.index_put((ids,), new)

    if params.presence_penalty != 0.0 or params.frequency_penalty != 0.0:
        if state.generated_counts:
            ids = torch.tensor(sorted(state.generated_counts), device=device, dtype=torch.long)
            counts = torch.tensor([state.generated_counts[i] for i in sorted(state.generated_counts)],
                                  device=device, dtype=torch.float32)
            delta = torch.zeros_like(counts)
            if params.frequency_penalty != 0.0:
                delta = delta + params.frequency_penalty * counts
            if params.presence_penalty != 0.0:
                delta = delta + params.presence_penalty
            row = row.index_put((ids,), row.index_select(0, ids) - delta)

    return row


def _filter_and_probs(row, params: SamplingParams):
    """温度 → top-k → softmax → top-p。返回 (probs, keep_mask)，都按**原始词表顺序**。"""
    vocab = row.shape[-1]
    scaled = row / params.temperature

    if params.top_k > 0 and params.top_k < vocab:
        # 并列分让**较小 token id 排在前面**：稳定降序排序会保留原始次序
        # （原始次序就是 token id 升序），于是前 k 个正好是「分数降序、同分取小 id」
        order = torch.argsort(scaled, descending=True, stable=True)
        keep_k = torch.zeros(vocab, dtype=torch.bool, device=scaled.device)
        keep_k[order[:params.top_k]] = True
        scaled = torch.where(keep_k, scaled, float("-inf"))

    probs = torch.softmax(scaled, dim=-1)

    if params.top_p < 1.0:
        # 从大到小累计，保留「刚刚跨过阈值」的那个：等价于「该 token 之前的累计 < p」
        order = torch.argsort(probs, descending=True, stable=True)
        sorted_probs = probs[order]
        cumulative = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        keep_sorted = cumulative < params.top_p
        # **至少保留一个**（需求 §1 的硬要求）。除了「刚刚跨过阈值」那个本来就会被留下，
        # 这里还兜住一种极端情况：比 float32 最小次正规数还小的 top_p（如 1e-300）
        # 与 float32 张量比较时会下溢成 0，于是 0 < 0 成立不了，集合会空掉。
        keep_sorted[0] = True
        keep = torch.zeros(vocab, dtype=torch.bool, device=probs.device)
        keep[order[keep_sorted]] = True
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        total = probs.sum()
        probs = probs / total
    return probs


class TorchSampler:
    """参考后端：全部用 Torch 算子，明确、好对照。"""

    name = "torch"

    def select(self, row, params: SamplingParams, state: SamplingState):
        """row 已经是施加过惩罚的 FP32 副本；返回一个 tensor 标量（留在 GPU 上）。"""
        if params.is_greedy:
            # 并列取最小 token id：torch.argmax 返回第一个最大值，正好是下标最小的
            return torch.argmax(row)
        probs = _filter_and_probs(row, params)
        return torch.multinomial(probs, num_samples=1, generator=state.generator).squeeze(0)

    def select_batch(self, rows, params_list, states):
        """整批选 token。两个后端用同一个接口，Engine 只认这个。

        返回与 rows 一一对应的 token id 列表，元素留在 GPU 上——
        调用方最后整批回传，不逐请求 .item()。
        """
        return [self.select(row, params, state)
                for row, params, state in zip(rows, params_list, states)]
