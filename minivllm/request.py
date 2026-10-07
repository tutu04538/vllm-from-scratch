"""`Request`：**服务端认可的请求状态**，由 Scheduler 持有（对应 vLLM `v1/request.py`）。

它**不是模型参数包**。step56 的 `SequenceConfig` 同时挂着 KV 缓存进度（`cache.length` /
`cache.block_table`）、GPU 张量、generator、draft 池……于是"谁改了这份状态"说不清：模型在写、
Scheduler 在读。这一版按 vLLM 切开：

| 字段 | 谁写 | 说明 |
|---|---|---|
| `prompt_token_ids` / `_all_token_ids` / `_output_token_ids` | 只有 `append_output_token_ids()` | 提交时复制 prompt，之后只增不改 |
| `num_computed_tokens` | **Scheduler**（`_update_after_schedule` 推进、`update_from_output` 修正） | 不是"显存里已写完的 token 数"，而是"已安排计算的" |
| `spec_token_ids` | Scheduler（57E） | 下一轮可能验证的草稿，尚未提交 |
| `status` / `stop_reason` | Scheduler | 状态与原因分开 |
| `block_hashes` / `_block_hasher` | Request 自己（57C 给出真实 hasher） | 已确定 token 的完整块 hash 链 |

**不放**：`cache.length`、`draft_cache`、GPU tensor、`torch.Generator`、抢占阻塞链、
counter RNG 事件数。这些要么属于 Scheduler 的计划（`num_computed_tokens`），要么属于执行侧
（KV 物理块、随机流），跨边界时只传快照。
"""

import enum
import time
from collections.abc import Sequence


class ReadOnlyTokenList(Sequence):
    """token 列表的**只读视图**：不复制底层 list，也不提供 append / extend。

    为什么不用 `list(...)` / `tuple(...)` 做只读：那会把整段历史复制一遍，而跨边界的快照
    每轮都要取。这里只包一层引用，索引/切片/len 直接落到原 list 上。

    step56 的 `ReadOnlyTokenList` 是同一个东西（第五十一关为了增量维护历史做的），这里照抄
    它的形状：对外尽量像 list，改内容只有一条路——`Request.append_output_token_ids()`。
    """

    __slots__ = ("_backing",)

    def __init__(self, backing: list):
        self._backing = backing

    def __getitem__(self, index):
        return self._backing[index]

    def __len__(self) -> int:
        return len(self._backing)

    def __iter__(self):
        return iter(self._backing)

    def __contains__(self, item) -> bool:
        return item in self._backing

    def __eq__(self, other) -> bool:
        if isinstance(other, ReadOnlyTokenList):
            other = other._backing
        return self._backing == other

    def __add__(self, other):
        return self._backing + list(other)

    def __radd__(self, other):
        return list(other) + self._backing

    def __repr__(self) -> str:
        return f"ReadOnlyTokenList({self._backing!r})"


class RequestStatus(enum.IntEnum):
    """请求状态。**PREEMPTED 之后都算已结束**（与 vLLM 同一条约定）。"""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # 下面都算已结束
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus"):
        from .outputs import FinishReason

        return _FINISHED_REASON.get(status)


_FINISHED_REASON = None  # 延迟到下面填，避免 outputs 与本模块互相 import


def _install_finished_reason_map() -> None:
    from .outputs import FinishReason

    global _FINISHED_REASON
    _FINISHED_REASON = {
        RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
        RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
        RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
        RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    }


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        arrival_time: float | None = None,
        priority: int = 0,
        block_hasher=None,
    ) -> None:
        self.request_id = request_id
        self.priority = priority
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time if arrival_time is not None else time.time()
        self.status = RequestStatus.WAITING
        self.stop_reason: int | str | None = None

        # `max_tokens` 是"要生成多少"，供调度与停止检查用；总上下文上限在 ModelConfig
        self.max_tokens = sampling_params.max_tokens

        # 提交时**复制**：用户在 add_request 之后继续改自己那份列表，不该影响运行中的请求
        self.prompt_token_ids = list(prompt_token_ids)
        self.num_prompt_tokens = len(self.prompt_token_ids)

        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = list(self.prompt_token_ids)

        self.spec_token_ids: list[int] = []
        # 70 关（异步调度）：**暂时预留的输出位置**个数。异步路径下 Scheduler 不等这一轮的
        # GPU 结果就排下一轮，于是先乐观地按"最多 K+1 个新 token"占位；结果回来后再按实际
        # 长度减掉。它**不是**用户可见的 token，也不算进 `num_tokens`——凡是要"已确认的
        # token"的地方（停止判定、prefix 发布、用户输出）都必须把它排除。
        self.num_output_placeholders: int = 0
        # 70 关：抢占会把这个请求**在飞的**输出标记成 stale（结果照常交付，但不许再改计数）。
        # 每轮在结果回来时按"这一轮排了多少行"抵扣，抵完为止（上游 `num_stale_output_tokens`）。
        self.num_stale_output_tokens: int = 0
        self.num_in_flight_tokens: int = 0
        self.num_computed_tokens = 0
        # 派生判断：这一轮还没算到已有历史的末尾（中间 prefill 块）。由 Scheduler 在
        # `_update_after_schedule()` 里更新。
        self.is_prefill_chunk = False
        self.num_preemptions = 0
        self.cache_salt: str | None = None

        # 68 关：结构化输出的请求级状态（grammar 的 FSM 就挂在它上面）。`None` = 不受约束。
        # 与上游同一条边界：**采样参数里有没有约束**决定它，而不是"引擎开没开结构化输出"。
        from .structured_output.request import StructuredOutputRequest

        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params)

        # 只读视图：防止外部直接 append（那样会绕过 append_output_token_ids 的同步）
        self.output_token_ids = ReadOnlyTokenList(self._output_token_ids)
        self.all_token_ids = ReadOnlyTokenList(self._all_token_ids)

        # 已确定 token 的完整块 hash 链。57A 默认没有 hasher（前缀缓存属 57C），
        # 所以这里是空的；接口留好，57C 接上真实 hasher 时调用点不变。
        self.block_hashes: list = []
        self._block_hasher = block_hasher
        self.update_block_hashes()

    @classmethod
    def from_engine_core_request(cls, request, block_hasher=None) -> "Request":
        """`EngineCoreRequest`（API 数据）→ `Request`（内部状态）。列表在这里复制。"""
        return cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            arrival_time=request.arrival_time,
            priority=request.priority,
            block_hasher=block_hasher,
        )

    # -------- 进度视图 --------

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def use_structured_output(self) -> bool:
        """这条请求要不要受语法约束（上游同名属性）。

        Scheduler 用它决定"要不要给它填掩码、要不要在提交后推进 grammar"；`Request` 自己
        不推进 FSM——状态归 `structured_output_request.grammar`，推进时机归 Scheduler
        （068 §2：不要让 proposer 永久推进 grammar）。
        """
        return self.structured_output_request is not None

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self):
        return RequestStatus.get_finished_reason(self.status)

    # -------- 唯一写入点 --------

    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        """已提交输出的**唯一**写入点：同步更新 `_output_token_ids` 与 `_all_token_ids`。

        两份列表都留（vLLM 也同时维护它们）：输出列表直接服务用户，完整列表服务"下一轮
        要算哪段"。只在这一个方法里同步，就不会出现两者对不上的状态。
        """
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
        self.update_block_hashes()

    def attach_block_hasher(self, block_hasher) -> None:
        """挂上块 hash 计算器（Scheduler 在请求进入时挂，见 `core/sched/scheduler.py`）。

        允许后挂是因为 hash 计算器属于**控制面**（它要 block_size 与是否开前缀缓存），
        而 `Request` 只负责"什么时候该算"。挂上时立刻补算已有历史——否则此前已经确定
        的完整块会**永久**缺少 hash，那部分前缀就再也命不中了。
        """
        if self._block_hasher is block_hasher:
            return
        self._block_hasher = block_hasher
        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """给新凑满的完整块补 hash（没有挂计算器时是空操作，例如关掉前缀缓存）。"""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    def __lt__(self, other: "Request") -> bool:
        """优先级队列的排序依据：数值小的 priority 更优先；同级按到达时间，再按 ID。

        （vLLM 的 `__lt__` 就是这三段；本关不用 id(self) 兜底——ID 已经唯一。）
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        return self.request_id < other.request_id

    def __repr__(self) -> str:
        return (f"Request({self.request_id!r}, status={self.status.name}, "
                f"tokens={self.num_tokens}, computed={self.num_computed_tokens}, "
                f"output={self.num_output_tokens})")


_install_finished_reason_map()
