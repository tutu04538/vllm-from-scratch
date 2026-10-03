"""`Sampler`：从 logits 到 token（对应 vLLM `v1/sample/sampler.py`）。

**它不负责请求生命周期**：不接 Request、不追加输出、不释放块、不决定 finished（199 §2）。
输入是 `[num_rows, vocab]` 的 logits 和按行组织的 `SamplingMetadata`，输出是
`SamplerOutput`（`sampled_token_ids`，形状 `[num_rows, 1]`）。谁把结果变成"请求的输出"、
谁判断停止，是 Runner 与 Scheduler 的事。

### 顺序（顺序本身就是语义）

```text
logits → fp32
→ 惩罚（repetition / frequency / presence）
→ 会改变 argmax 的约束（min_tokens 屏蔽停止 token）
→ greedy 行：argmax            ← 不除温度（温度 0 除以它会得到 inf/nan）
→ random 行：除温度 → top-k/top-p → 指数竞赛抽样
→ 用一个 where 把两路合起来（temperature < eps 的行取 greedy 的结果）
```

与 57D 相比，59 关把**惩罚从 `sample()` 挪进了 `apply_logits_processors()`**：上游就是在
`apply_logits_processors` 里做"白名单 → bad words → 非 argmax 不变处理器 → 惩罚"这一串，
`sample()` 只负责采样。两处顺序对结果没有影响（`-inf` 掩码与加性惩罚可交换），挪过来是为了
让投机路径复用同一条"先改 logits、再采样"的边界（059 §3.2）。

为什么"改变 argmax 的约束"与惩罚要在 greedy 之前：贪心行也要受它们影响——
`min_tokens` 的意义就是"还不许吐出停止 token"，贪心行当然也算。

### min_tokens 是"暂不允许采到什么"，不是"结束请求"

同一件事有两处判断，职责不同（199 §2）：

    这里（logits 处理）  还没生成够的行，把它的停止 token 打成 -inf
    Scheduler.check_stop 已经**提交**的 token 是不是停止 token → 决定结束

前者管"采不出来"，后者管"采到了之后怎么收尾"。两处都不做对方的判断：采样器不决定请求
生死，调度器也不去改 logits。

### 投机相关的两个入口（059 §2）

    predict_bonus_token=True        这一行是 bonus 行 → 惩罚的历史要把**全部草稿**算进去
                                    （能走到 bonus 就说明草稿都被接受了）
    apply_min_tokens_for_spec_decode 验证行是"每请求 K_i 行"，屏蔽的是**前 n_mask 行**
                                    （第 j 行面对的历史是"已提交 + 草稿前缀 [:j]"）
"""

import numpy as np
import torch

from ..outputs import SamplerOutput
from .metadata import SAMPLING_EPS, SamplingMetadata
from .ops.penalties import apply_all_penalties
from .ops.topk_topp_sampler import TopKTopPSampler


class Sampler:
    def __init__(self) -> None:
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata,
                predict_bonus_token: bool = False) -> SamplerOutput:
        # 断言两条路不会同时成立：全贪心与全随机只在"批为空"时才同时为真
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        logits = logits.to(torch.float32)
        logits = self.apply_logits_processors(logits, sampling_metadata, predict_bonus_token)
        sampled = self.sample(logits, sampling_metadata)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(dim=1))

    # -------- 1) 改 logits：惩罚 + 会改变 argmax 的约束 --------

    def apply_logits_processors(self, logits: torch.Tensor,
                                sampling_metadata: SamplingMetadata,
                                predict_bonus_token: bool = False) -> torch.Tensor:
        """原地应用逐行约束（顺序与上游一致：惩罚 → min_tokens）。

        vLLM 在这里还有白名单（allowed_token_ids）、bad words 与一个插件框架；本关只有
        惩罚与 min_tokens 两条。

        `predict_bonus_token=True` 时，惩罚用的历史要换成"已提交 + 全部草稿"（上游同名参数）：
        这是投机路径采样 bonus token 的那一行。
        """
        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token and not sampling_metadata.no_penalties:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids, sampling_metadata.spec_token_ids)
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        logits = self.apply_min_tokens(logits, sampling_metadata)
        return logits

    def apply_penalties(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata,
                        output_token_ids: list[list[int]] | None = None) -> torch.Tensor:
        """三种惩罚（上游 `Sampler.apply_penalties` / 这里的内联版本）。

        整批都没人用惩罚时直接返回（`no_penalties`）；逐行的惩罚值仍然是张量。
        """
        if sampling_metadata.no_penalties:
            return logits
        if output_token_ids is None:
            output_token_ids = sampling_metadata.output_token_ids
        return apply_all_penalties(
            logits, sampling_metadata.prompt_token_ids, output_token_ids,
            sampling_metadata.presence_penalties, sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties)

    @staticmethod
    def _combine_outputs_with_spec_tokens(
            output_token_ids: list[list[int]],
            spec_token_ids: list[list[int]] | None = None) -> list[list[int]]:
        """bonus 行的历史 = 已提交历史 + **全部**草稿（行数不变，逐请求）。

        对应上游 `Sampler._combine_outputs_with_spec_tokens`——注意它与
        `RejectionSampler._combine_outputs_with_spec_tokens` **不是**同一个东西：那边按草稿
        前缀展开成 K 行（验证行），这边一行一条请求（bonus 行）。
        """
        if spec_token_ids is None:
            return output_token_ids

        return [
            [*out, *spec] if spec else out
            for out, spec in zip(output_token_ids, spec_token_ids)
        ]

    def apply_min_tokens(self, logits: torch.Tensor,
                         sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """min_tokens：还没生成够的行，把它的停止 token 打成 -inf（上游 `MinTokensLogitsProcessor.apply`）。

        用**一次** `index_put_` 处理整批：逐行 `logits[row, stop_ids] = -inf` 每行都是一次
        H2D 拷贝，投机路径的 bonus 行会白付 B 次。
        """
        rows, tokens = [], []
        for row, (min_tokens, stop_ids) in enumerate(zip(sampling_metadata.min_tokens,
                                                        sampling_metadata.stop_token_ids)):
            if not stop_ids or len(sampling_metadata.output_token_ids[row]) >= min_tokens:
                continue
            rows.extend([row] * len(stop_ids))
            tokens.extend(stop_ids)
        return self._mask_tokens(logits, rows, tokens)

    def apply_min_tokens_for_spec_decode(
            self, logits: torch.Tensor, sampling_metadata: SamplingMetadata,
            num_draft_tokens: list[int]) -> torch.Tensor:
        """投机版 min_tokens 屏蔽（对应上游 `MinTokensLogitsProcessor.apply_with_spec_decode`）。

        验证行是"每请求 K_i 行"，第 j 行面对的历史是"已提交 + 草稿前缀 `[:j]`"，长度
        `len(已提交输出) + j`；所以只要 `len(out) + j < min_tokens` 就该屏蔽，也就是**前
        `n_mask = clamp(min_tokens - len(out), 0, K_i)` 行**。上游的算例：
        `num_draft_tokens=[2,3,1]` → `logits` 是 6 行，`cumsum=[0,2,5,6]`。

        没有草稿的请求（K=0）不占验证行，`n_mask` 也是 0，自然跳过。
        """
        if not sampling_metadata.min_tokens:
            return logits

        num_draft_arr = np.array(num_draft_tokens, dtype=np.int64)
        cumsum = np.concatenate([[0], np.cumsum(num_draft_arr)])

        all_rows: list[np.ndarray] = []
        all_tokens: list[np.ndarray] = []
        for req_index, (min_tokens, stop_ids) in enumerate(zip(
                sampling_metadata.min_tokens, sampling_metadata.stop_token_ids)):
            if not stop_ids:
                continue
            remaining = min_tokens - len(sampling_metadata.output_token_ids[req_index])
            n_mask = int(min(max(remaining, 0), num_draft_arr[req_index]))
            if n_mask <= 0:
                continue
            offset = int(cumsum[req_index])
            row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
            all_rows.append(np.repeat(row_indices, len(stop_ids)))
            all_tokens.append(np.tile(np.array(stop_ids, dtype=np.int64), n_mask))

        if not all_rows:
            return logits
        return self._mask_tokens(logits, np.concatenate(all_rows),
                                 np.concatenate(all_tokens))

    @staticmethod
    def _mask_tokens(logits: torch.Tensor, rows, tokens) -> torch.Tensor:
        """把 `(row, token)` 位置打成 -inf（空集是空操作）。"""
        if len(rows) == 0:
            return logits
        index = (torch.as_tensor(rows, dtype=torch.int64, device=logits.device),
                 torch.as_tensor(tokens, dtype=torch.int64, device=logits.device))
        neg_inf = torch.tensor(-float("inf"), dtype=logits.dtype, device=logits.device)
        logits.index_put_(index, neg_inf)
        return logits

    # -------- 2) 采样 --------

    def sample(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """返回 `[num_rows]` 的 token。greedy 行与 random 行在同一批里各走各的。

        惩罚不在这里做（已由 `apply_logits_processors` 施加）。
        """
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
