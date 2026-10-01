"""`Sampler`：从 logits 到 token（对应 vLLM `v1/sample/sampler.py`）。

**它不负责请求生命周期**：不接 Request、不追加输出、不释放块、不决定 finished（199 §2）。
输入是 `[num_rows, vocab]` 的 logits 和按行组织的 `SamplingMetadata`，输出是
`SamplerOutput`（`sampled_token_ids`，形状 `[num_rows, 1]`）。谁把结果变成"请求的输出"、
谁判断停止，是 Runner 与 Scheduler 的事。

### 顺序（顺序本身就是语义）

```text
logits → fp32
→ 会改变 argmax 的约束（min_tokens 屏蔽停止 token）
→ 惩罚（repetition / frequency / presence）
→ greedy 行：argmax            ← 不除温度（温度 0 除以它会得到 inf/nan）
→ random 行：除温度 → top-k/top-p → 指数竞赛抽样
→ 用一个 where 把两路合起来（temperature < eps 的行取 greedy 的结果）
```

为什么"改变 argmax 的约束"必须在 greedy 之前：贪心行也要受惩罚和 min_tokens 影响——
`min_tokens` 的意义就是"还不许吐出停止 token"，贪心行当然也算。

### min_tokens 是"暂不允许采到什么"，不是"结束请求"

同一件事有两处判断，职责不同（199 §2）：

    这里（logits 处理）  还没生成够的行，把它的停止 token 打成 -inf
    Scheduler.check_stop 已经**提交**的 token 是不是停止 token → 决定结束

前者管"采不出来"，后者管"采到了之后怎么收尾"。两处都不做对方的判断：采样器不决定请求
生死，调度器也不去改 logits。
"""

import torch

from ..outputs import SamplerOutput
from .metadata import SAMPLING_EPS, SamplingMetadata
from .ops.penalties import apply_all_penalties
from .ops.topk_topp_sampler import TopKTopPSampler


class Sampler:
    def __init__(self) -> None:
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> SamplerOutput:
        # 断言两条路不会同时成立：全贪心与全随机只在"批为空"时才同时为真
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        logits = logits.to(torch.float32)
        logits = self.apply_logits_processors(logits, sampling_metadata)
        sampled = self.sample(logits, sampling_metadata)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(dim=1))

    # -------- 1) 会改变 argmax 的约束 --------

    def apply_logits_processors(self, logits: torch.Tensor,
                                sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """原地应用"非 argmax 不变"的约束：目前只有 min_tokens 的停止 token 屏蔽。

        vLLM 在这里还有白名单（allowed_token_ids）、bad words 与一个插件框架；本关只有
        min_tokens 一条，所以直接写在这里，并把它的语义写清楚。
        """
        if not sampling_metadata.min_tokens:
            return logits
        for row, (min_tokens, stop_ids) in enumerate(zip(sampling_metadata.min_tokens,
                                                        sampling_metadata.stop_token_ids)):
            if not stop_ids or len(sampling_metadata.output_token_ids[row]) >= min_tokens:
                continue
            # 还没生成够 → 这一行的停止 token 全部屏蔽
            logits[row, stop_ids] = -float("inf")
        return logits

    # -------- 2) 采样 --------

    def sample(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """返回 `[num_rows]` 的 token。greedy 行与 random 行在同一批里各走各的。"""
        if not sampling_metadata.no_penalties:
            logits = apply_all_penalties(
                logits, sampling_metadata.prompt_token_ids, sampling_metadata.output_token_ids,
                sampling_metadata.presence_penalties, sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties)

        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        temperature = sampling_metadata.temperature
        if temperature is None:
            raise ValueError("批里有随机行，却没有温度张量（SamplingMetadata 构建错了）")
        # 温度低的行走贪心，这里先把它们的温度换成 1.0，避免除以 0（与 vLLM 同款处理）
        safe_temperature = torch.where(temperature < SAMPLING_EPS,
                                      torch.ones_like(temperature), temperature)
        logits = logits.div_(safe_temperature.unsqueeze(dim=1))

        random_sampled = self.topk_topp_sampler(logits, sampling_metadata.generators,
                                                sampling_metadata.top_k, sampling_metadata.top_p)
        if greedy_sampled is None:
            return random_sampled
        return torch.where(temperature < SAMPLING_EPS, greedy_sampled, random_sampled)

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        """argmax。**不除温度**：温度 0 的走这条路，除下去会得到 inf/nan。"""
        return logits.argmax(dim=-1).view(-1)
