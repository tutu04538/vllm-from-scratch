"""`Sampler`：从 logits 到 token（对应 vLLM `v1/sample/sampler.py`）。

**它不负责请求生命周期**：不接 Request、不追加输出、不释放块、不决定 finished（199 §2）。
输入是 `[num_rows, vocab]` 的 logits 和按行组织的 `SamplingMetadata`，输出是
`SamplerOutput`（`sampled_token_ids`，形状 `[num_rows, 1]`；+ 可选的 logprobs）。谁把结果
变成"请求的输出"、谁判断停止，是 Runner 与 Scheduler 的事。

### 顺序（顺序本身就是语义）

```text
（要 logprobs 且模式是 raw_*：先把"原始"那一份留下来）
logits → fp32
→ 惩罚（repetition / frequency / presence）
→ 会改变 argmax 的约束（min_tokens 屏蔽停止 token）
→ greedy 行：argmax            ← 不除温度（温度 0 除以它会得到 inf/nan）
→ random 行：除温度 → min_p → top-k/top-p → 指数竞赛抽样
→ 用一个 where 把两路合起来（temperature < eps 的行取 greedy 的结果）
→（要 logprobs：按模式交付 raw 或 processed 的那一份，取 top-k + 实际采到的 token）
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

### logprobs 的四种模式（68 关；对应上游 `ModelConfig.logprobs_mode`）

    raw_logprobs         logits 的 log_softmax（**惩罚/温度之前**，但**包含语法掩码**）
    raw_logits           原始 logits 本身（需要未归一化分数时用）
    processed_logprobs   惩罚 + 温度 + min_p + top-k/top-p **之后**的 log_softmax
    processed_logits     上面那份 logits 本身

"包含语法掩码"这条很关键：结构化输出的掩码是在 Runner 里**先**打到 logits 上、再进采样器的
（上游同序，见 `structured_output/utils.py::apply_grammar_bitmask`），所以两份都带着掩码，
不会出现"交付的 logprobs 认为某个非法 token 还有概率"这种事。
对 greedy 行，`processed_*` 只包含惩罚与掩码（温度/top-k/top-p 对贪心行不施加）。
"""

import numpy as np
import torch

from ..outputs import LogprobsTensors, SamplerOutput
from .metadata import SAMPLING_EPS, SamplingMetadata
from .ops.penalties import apply_all_penalties
from .ops.topk_topp_sampler import TopKTopPSampler


class Sampler:
    def __init__(self, logprobs_mode: str = "raw_logprobs") -> None:
        self.logprobs_mode = logprobs_mode
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode)

    def forward(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata,
                predict_bonus_token: bool = False,
                logprobs_mode_override: str | None = None) -> SamplerOutput:
        """采样一行或多行，可选交付 logprobs（68 关加的两个参数，上游同签名）。

        `logprobs_mode_override`：只给"拒绝采样器里的 bonus 行"用——那里的 bonus 采样
        只是为了拿到 bonus token，它的 logprobs 要到 `_get_logprobs_tensors` 里与候选行
        拼在一起才算，所以上游强制它返回 **logits**（而不是 logprobs），本仓库同款。
        """
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # 断言两条路不会同时成立：全贪心与全随机只在"批为空"时才同时为真
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)

        # 原始那一份要在**任何原地改动之前**留下来（上游同款注释：top-k logprobs 用的是
        # 未施加惩罚/温度的 logits）。只有要 logprobs 且模式是 raw_* 时才留。
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None
        if num_logprobs is not None:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                raw_logprobs = (logits.clone() if logits.dtype == torch.float32
                                else logits.to(torch.float32))

        logits = logits.to(torch.float32)
        logits = self.apply_logits_processors(logits, sampling_metadata, predict_bonus_token)
        # **不把 override 传给 `sample()`**（上游同款）：override 只决定"开头要不要留 raw 那一份"，
        # `sample()` 里用的是引擎级模式。拒绝采样器的 bonus 行因此会经历"两次 log_softmax"
        # （processed_logprobs 模式下），但 `log_softmax` 幂等，交付值逐值不变
        # （2026-10-06 独立复核；见 docs/step68_alignment.md §3.5）。
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            # processed_* 模式下"要交付的那一份"是采样时才算出来的，直接顶替
            raw_logprobs = processed_logprobs
        sampled = sampled.long()

        if num_logprobs is None:
            logprobs_tensors = None
        elif num_logprobs == -1:
            # 全词表：不排序也不排名次，直接把整份分布交出去（上游同款）
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0))
        else:
            logprobs_tensors = self.gather_logprobs(raw_logprobs, num_logprobs, sampled)

        return SamplerOutput(
            sampled_token_ids=sampled.to(torch.int32).unsqueeze(dim=1),
            logprobs_tensors=logprobs_tensors)

    # -------- 1) 改 logits：惩罚 + 会改变 argmax 的约束 --------

    def apply_logits_processors(self, logits: torch.Tensor,
                                sampling_metadata: SamplingMetadata,
                                predict_bonus_token: bool = False) -> torch.Tensor:
        """原地应用逐行约束（顺序与上游一致：惩罚 → min_tokens）。

        vLLM 在这里还有白名单（allowed_token_ids）、bad words、logit_bias 与一个插件框架；
        本仓库只有惩罚与 min_tokens 两条，其余在请求期明确拒绝（068 §3.6 的三态矩阵）。

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

    def sample(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata,
               logprobs_mode: str | None = None
               ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """返回 `(token, 要交付的 processed 那一份)`。greedy 行与 random 行在同一批里各走各的。

        惩罚不在这里做（已由 `apply_logits_processors` 施加）。

        第二个返回值只在 `processed_*` 模式下非 None；语义与上游 `Sampler.sample` 的二元组
        一致（上游把它叫 `processed_logprobs`，但 `processed_logits` 模式下装的是 logits）。
        """
        mode = logprobs_mode or self.logprobs_mode
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                # 全贪心：温度/top-k/top-p 都不施加，所以 processed 那一份就是**当前**的 logits
                processed = None
                if sampling_metadata.max_num_logprobs is not None:
                    if mode == "processed_logits":
                        processed = logits
                    elif mode == "processed_logprobs":
                        processed = self.compute_logprobs(logits)
                return greedy_sampled, processed

        temperature = sampling_metadata.temperature
        if temperature is None:
            raise ValueError("批里有随机行，却没有温度张量（SamplingMetadata 构建错了）")
        # 温度低的行走贪心，这里先把它们的温度换成 1.0，避免除以 0（与 vLLM 同款处理）
        safe_temperature = torch.where(temperature < SAMPLING_EPS,
                                      torch.ones_like(temperature), temperature)
        logits = logits.div_(safe_temperature.unsqueeze(dim=1))

        # min_p：上游把它做成"不改 argmax"的 logits processor（`MinPLogitsProcessor.apply`），
        # 位置就在温度之后、top-k/top-p 之前。算式逐行照抄：
        #     adjusted = max(softmax(logits)) * min_p;  prob < adjusted 的全部 -inf
        # 本项目没有 logits processor 插件框架（三态矩阵里明写），所以内联同一算式。
        logits = self.apply_min_p(logits, sampling_metadata.min_p)

        random_sampled, processed = self.topk_topp_sampler(
            logits, sampling_metadata.generators,
            sampling_metadata.top_k, sampling_metadata.top_p)
        if greedy_sampled is None:
            return random_sampled, processed
        sampled = torch.where(temperature < SAMPLING_EPS, greedy_sampled, random_sampled)
        return sampled, processed

    @staticmethod
    def apply_min_p(logits: torch.Tensor, min_p: torch.Tensor | None) -> torch.Tensor:
        """min_p 屏蔽（上游 `v1/sample/logits_processor/builtin.py::MinPLogitsProcessor.apply`）。

        `min_p=None` 表示整批都没人用它 —— 与上游"`min_p_count == 0` 直接返回"同一条路径。
        """
        if min_p is None:
            return logits
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        adjusted_min_p = max_probabilities.mul_(min_p.unsqueeze(dim=1))
        invalid_token_mask = probability_values < adjusted_min_p
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        """argmax。**不除温度**：温度 0 的走这条路，除下去会得到 inf/nan。"""
        return logits.argmax(dim=-1).view(-1)

    # -------- 3) logprobs（68 关）--------

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        """log_softmax（上游同名静态方法）。"""
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """每行里"大于等于 values"的个数（上游 `sample/ops/logprobs.py` 同名函数）。

        它就是选中 token 的**名次**（1 = 概率最大）：比它大的都排在前面，加上它自己。
        上游用 `torch.compile` 包一层省显存；本仓库不引入编译（69 关才碰 CUDA Graph），
        直接写同一算式——数值逐位相同，差的是编译开销。
        """
        torch._check(x.shape[0] >= 1)
        torch._check(x.shape[0] == values.shape[0])
        return (x >= values).sum(-1)

    @staticmethod
    def gather_logprobs(logprobs: torch.Tensor, num_logprobs: int,
                        token_ids: torch.Tensor) -> LogprobsTensors:
        """取"前 `num_logprobs` 名 + 实际采到的那个 token"（上游同名静态方法，逐行对应）。

        第 0 列永远是**采到的 token**（`token_ids`），后面才是 top-k：这样输出处理只需读
        第 0 列就知道"这个位置交付了哪个 token、它的 logprob 是多少"。选中 token 同时也在
        top-k 里时，字典合并天然只留一份。

        `token_ids` 必须是 int64（上游同款断言：int32 在 `gather` 里会被当成别的语义）。
        """
        assert token_ids.dtype == torch.int64
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)
        token_ranks = Sampler.batched_count_greater_than(logprobs, token_logprobs)

        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
        return LogprobsTensors(indices.to(torch.int32), logprobs, token_ranks.to(torch.int32))
