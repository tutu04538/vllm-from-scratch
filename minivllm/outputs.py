"""三层输出/输入协议（对应 vLLM 的 `v1/engine/__init__.py` 与 `v1/outputs.py`）。

数据在三个边界上流动，**每层的名字与"谁拥有"都不同**：

    API/用户                  EngineCore                  执行侧（Worker/Runner）
    ─────────                 ───────────                  ──────────────────────
    EngineCoreRequest    →    Request（Scheduler 持有）  →  NewRequestData / CachedRequestData
    RequestOutput        ←    EngineCoreOutput           ←  ModelRunnerOutput

- `EngineCoreRequest`：API 交给 EngineCore 的**数据**（不是内部 Request）。可变列表要复制。
- `ModelRunnerOutput`：执行侧交回的结果。**行号 ≠ 请求 ID**：`req_id_to_index` 是唯一的
  映射依据，Scheduler 不能 `zip(running, sampled_token_ids)`。
- `EngineCoreOutput`：本轮**最终提交**的 token 与结束原因（不是原始采样结果）。
- `RequestOutput`：用户可见的累计结果，由 `OutputProcessor` 产出。

对应 vLLM：`EngineCoreRequest`（`v1/engine/__init__.py:100`）、`EngineCoreOutput`（同文件
`:189`）、`EngineCoreOutputs`（`:242`）、`ModelRunnerOutput`（`v1/outputs.py:310`）、
`RequestOutput`（`vllm/outputs.py`）。vLLM 用 msgspec.Struct 是为了跨进程零拷贝，本关同进程，
用 dataclass 就够——**这不是"等价实现"，只是本关子集**。
"""

import enum
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

from .logprobs import SampleLogprobs
from .sampling_params import SamplingParams


class FinishReason(enum.IntEnum):
    """结束原因。与 `RequestStatus` 分开：状态说"现在处于什么阶段"，原因说"为什么结束"。

    对应 vLLM `v1/engine/__init__.py::FinishReason`（那里还有 REPETITION 等）。
    """

    STOP = 0        # 遇到 eos / stop token
    LENGTH = 1      # 达到 max_tokens 或上下文上限
    ABORT = 2       # 被调用方中止
    ERROR = 3       # 内部错误


@dataclass
class EngineCoreRequest:
    """API → EngineCore 的输入数据。

    字段与 195 §4 一致。`prompt_token_ids` 由 `Request.from_engine_core_request()` 复制，
    避免用户在提交之后继续 `prompt.append(...)` 改掉运行中的请求。
    """

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float
    priority: int = 0
    cache_salt: str | None = None


@dataclass
class EngineCoreOutput:
    """EngineCore → 上层：一条请求本轮**最终提交**的 token 与结束原因。

    注意不是"Runner 采到了什么"：被拒绝的草稿、EOS 之后被截掉的候选，都不会出现在这里
    （截断发生在 `Scheduler._update_request_with_output()` 里）。

    `new_logprobs`（68 关）与 `new_token_ids` **逐位置对齐**：长度就是 `len(new_token_ids)`
    （Scheduler 按提交后的数量切片，见 `LogprobsLists.slice_request`），所以被截断的尾巴
    在 logprobs 里也不会漏出来（068 §3.5）。
    """

    request_id: str
    new_token_ids: list[int]
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    new_logprobs: "LogprobsLists | None" = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class EngineCoreOutputs:
    """一轮的容器。本关单客户端，不需要 vLLM 的多 client-index 分桶。"""

    outputs: list[EngineCoreOutput] = field(default_factory=list)
    finished_requests: set[str] = field(default_factory=set)


@dataclass
class DraftTokenIds:
    """执行侧交回的**下一轮草稿**（对应 vLLM `v1/outputs.py::DraftTokenIds`）。

    只带请求 ID 与候选 token：CPU 侧的 Scheduler 只需要这些来决定下一轮的预算，
    **不需要**持有 `[P, V]` 的草稿概率（那是执行侧自己的东西，下一轮按实际采用的草稿
    前缀去组织 q，见 199 §5）。

    这里多带一个 `draft_probs`（vLLM 放在 Runner 里按请求 ID 存）：本关把概率与草稿一起
    交回，由 Runner 自己按行号组织，少一份跨结构的索引。
    """

    req_ids: list[str]
    draft_token_ids: list[list[int]]
    draft_probs: torch.Tensor | None = None
    # 60 关：GPU 提议者的输出是**固定宽度**的（`[B, K]`，尾部用 -1 占位），所以"宽度"不等于
    # "有几枚真草稿"——这个字段给出每行的**有效前缀长度**（CPU 提议者给 None = 每一枚都有效）。
    # Scheduler 在收下草稿时按它裁一遍（`update_scheduler_for_invalid_drafts`），
    # 保证哨兵 `-1` 不会变成真实 token。
    num_valid_draft_tokens: list[int] | None = None


@dataclass
class SamplerOutput:
    """`Sampler` 的产物：token（+ logprobs），对应 vLLM `v1/outputs.py::SamplerOutput`。

    形状：普通采样一行出一个 token（`[num_rows, 1]`）；投机是 `[B, max_spec_len + 1]`，
    被拒绝的位置填 `PLACEHOLDER_TOKEN_ID(-1)`（int32，上游同 dtype）。为什么不让采样器直接
    产出 `list[list[int]]`：它是**执行侧**的东西，行号与请求 ID 的对应关系是 Runner 才知道的
    事（见 `_bookkeeping_sync` / `RejectionSampler.parse_output`）。

    `logprobs_tensors`（68 关）与 `sampled_token_ids` 同源、同行序：普通采样是
    `[num_rows, k+1]`，投机是 `[num_tokens, k+1]`（每个**候选位**一行，无效位由
    `parse_output` 用同一张 valid_mask 过滤）。
    """

    sampled_token_ids: torch.Tensor
    logprobs_tensors: "LogprobsTensors | None" = None


@dataclass
class ModelRunnerOutput:
    """执行侧 → Scheduler 的结果。

    `sampled_token_ids` 与 `req_ids` 对齐：每项是该请求本轮返回的**候选输出序列**——
    普通生成 1 个、未完成的 prefill 是空列表、投机可能多个。

    **必须查 `req_id_to_index` 取结果**，不能按 Scheduler 自己的顺序 zip：Runner 允许
    重排紧凑 batch（本关的用例就故意把行反着返回）。

    `logprobs`（68 关）与 `req_ids` 同样对齐（`LogprobsLists.slice_request(req_index, n)`
    按请求切），`cu_num_generated_tokens` 给出每请求的行起始偏移——投机时一条请求一行可能
    交付 0~K+1 个位置，没有这个偏移就无法按请求切片。
    """

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]
    logprobs: "LogprobsLists | None" = None

    @classmethod
    def make_empty(cls) -> "ModelRunnerOutput":
        return cls(req_ids=[], req_id_to_index={}, sampled_token_ids=[])


@dataclass
class RequestOutput:
    """用户可见的结果：累计 token（不是增量），外加结束状态。`text` 由 tokenizer 提供。

    `stop_reason` 只在结束那一条上有值：显式 stop token 命中时是那个 token id，
    否则是 None（对应 vLLM `CompletionOutput.stop_reason`）。

    `logprobs` / `cumulative_logprob`（68 关）与 `token_ids` **逐位置对齐**：
    第 i 个 token 的 logprobs 就是 `logprobs[i]`（它是累计视图，不是这一轮的增量——和
    `token_ids` 一样）。`cumulative_logprob` 是"到目前所有生成 token 的 logprob 之和"，
    上游用它算 perplexity：
    `ppl = exp(-cumulative_logprob / len(token_ids))`。
    """

    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    finished: bool = False
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    text: str | None = None
    logprobs: SampleLogprobs | None = None
    cumulative_logprob: float | None = None


# ---------------------------------------------------------------------------
# logprobs 的两层容器（68 关；对应 vLLM `v1/outputs.py` 的同名 NamedTuple）
# ---------------------------------------------------------------------------


class LogprobsTensors(NamedTuple):
    """执行侧的 logprobs（**GPU 张量**，还没下 CPU）。

    形状（上游逐字）：
        logprob_token_ids     [num_positions, max_num_logprobs + 1]
        logprobs              [num_positions, max_num_logprobs + 1]
        selected_token_ranks  [num_positions]

    第 0 列永远是**实际采到的那个 token**（其余是 top-k），rank 是它在整份分布里的名次。
    投机时一个"位置"就是一轮里交付的每一个 token（接受的候选 / 恢复 token / bonus），
    拒绝掉的候选位在这里根本不存在——它们在 `parse_output` 里已被 `valid_mask` 滤掉。
    """

    logprob_token_ids: torch.Tensor
    logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor
    cu_num_generated_tokens: list[int] | None = None

    def tolists(self, cu_num_generated_tokens: list[int] | None = None) -> "LogprobsLists":
        """转成 CPU numpy（跨执行边界传的就是这一份）。"""
        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            cu_num_generated_tokens if cu_num_generated_tokens is not None
            else self.cu_num_generated_tokens,
        )

    def to_cpu_nonblocking(self) -> "LogprobsTensors":
        """非阻塞地搬到 CPU（上游同名方法；本仓库同进程，直接同步搬）。"""
        if self.logprob_token_ids.device.type == "cpu":
            return self
        return LogprobsTensors(
            self.logprob_token_ids.to("cpu", non_blocking=True),
            self.logprobs.to("cpu", non_blocking=True),
            self.selected_token_ranks.to("cpu", non_blocking=True),
            self.cu_num_generated_tokens,
        )

    def filter(self, mask: torch.Tensor) -> "LogprobsTensors":
        """按行掩码过滤（上游同名方法）。

        投机路径用它把"被拒绝的候选位"整行丢掉：`parse_output` 用**同一个** valid_mask
        既滤 token 又滤 logprobs，所以两边的位置永远对得上（068 §3.5）。
        """
        assert self.cu_num_generated_tokens is None, (
            "filter 不能与 cu_num_generated_tokens 一起用（上游同款断言）："
            "已经有每请求偏移的容器说明它已经被切开过")
        return LogprobsTensors(
            self.logprob_token_ids[mask],
            self.logprobs[mask],
            self.selected_token_ranks[mask],
        )

    @staticmethod
    def cat(tensors: list["LogprobsTensors"],
            cu_num_generated_tokens: list[int] | None = None) -> "LogprobsTensors":
        """把若干段 logprobs 拼起来（上游同名方法）。"""
        assert tensors
        assert cu_num_generated_tokens is not None or all(
            tensor.cu_num_generated_tokens is None for tensor in tensors)
        if len(tensors) == 1:
            tensor = tensors[0]
            if cu_num_generated_tokens is None:
                return tensor
            return tensor._replace(cu_num_generated_tokens=cu_num_generated_tokens)
        return LogprobsTensors(
            logprob_token_ids=torch.cat([t.logprob_token_ids for t in tensors]),
            logprobs=torch.cat([t.logprobs for t in tensors]),
            selected_token_ranks=torch.cat([t.selected_token_ranks for t in tensors]),
            cu_num_generated_tokens=cu_num_generated_tokens,
        )

    @staticmethod
    def empty_cpu(num_positions: int, num_tokens_per_position: int) -> "LogprobsTensors":
        """建一个空的 CPU 容器（上游同名方法；给"这条请求这一轮没有位置"占位用）。"""
        logprob_token_ids = torch.empty((num_positions, num_tokens_per_position),
                                        dtype=torch.int32, device="cpu")
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(num_positions, dtype=torch.int32, device="cpu")
        return LogprobsTensors(logprob_token_ids, logprobs, selected_token_ranks)


class LogprobsLists(NamedTuple):
    """CPU 侧的 logprobs（numpy），跨执行边界之后的那一份（上游同名 NamedTuple）。

    `cu_num_generated_tokens[i]` 是第 i 条请求在行方向上的**起始偏移**；投机时每请求
    交付的位置数不同（0 ~ K+1），所以切片必须用它，不能拿请求序号当行号。
    """

    logprob_token_ids: np.ndarray
    logprobs: np.ndarray
    sampled_token_ranks: np.ndarray
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int) -> "LogprobsLists":
        """切出第 `req_idx` 条请求的 `num_positions` 个位置（上游同名方法）。

        `num_positions` 用**已经提交**的 token 数（Scheduler 传 `len(new_token_ids)`）：
        停止 token 把这一轮剩下的候选截断时，logprobs 也同步截断——尾部不会漏出来。
        """
        if self.cu_num_generated_tokens is not None:
            req_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = req_idx + num_positions
        return LogprobsLists(
            self.logprob_token_ids[req_idx:end_idx],
            self.logprobs[req_idx:end_idx],
            self.sampled_token_ranks[req_idx:end_idx],
            None,
        )
