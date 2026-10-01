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

import torch

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
    """

    request_id: str
    new_token_ids: list[int]
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class EngineCoreOutputs:
    """一轮的容器。本关单客户端，不需要 vLLM 的多 client-index 分桶。"""

    outputs: list[EngineCoreOutput] = field(default_factory=list)
    finished_requests: set[str] = field(default_factory=set)


@dataclass
class SamplerOutput:
    """`Sampler` 的产物：**只有 token**，形状 `[num_rows, 1]`（对应 vLLM
    `v1/outputs.py::SamplerOutput` 的子集——它还有 logprobs 张量，本关不做 logprobs）。

    形状里那个 1 是"一行出一个 token"；投机（57E）会变成 `max_spec_len + 1`，
    被拒绝的位置填 -1。为什么不让采样器直接产出 `list[list[int]]`：它是**执行侧**的东西，
    行号与请求 ID 的对应关系是 Runner 才知道的事（见 `_bookkeeping_sync`）。
    """

    sampled_token_ids: torch.Tensor


@dataclass
class ModelRunnerOutput:
    """执行侧 → Scheduler 的结果。

    `sampled_token_ids` 与 `req_ids` 对齐：每项是该请求本轮返回的**候选输出序列**——
    普通生成 1 个、未完成的 prefill 是空列表、投机可能多个。

    **必须查 `req_id_to_index` 取结果**，不能按 Scheduler 自己的顺序 zip：Runner 允许
    重排紧凑 batch（本关的用例就故意把行反着返回）。
    """

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]

    @classmethod
    def make_empty(cls) -> "ModelRunnerOutput":
        return cls(req_ids=[], req_id_to_index={}, sampled_token_ids=[])


@dataclass
class RequestOutput:
    """用户可见的结果：累计 token（不是增量），外加结束状态。`text` 由 tokenizer 提供。

    `stop_reason` 只在结束那一条上有值：显式 stop token 命中时是那个 token id，
    否则是 None（对应 vLLM `CompletionOutput.stop_reason`）。
    """

    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    finished: bool = False
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    text: str | None = None
