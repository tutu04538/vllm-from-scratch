"""把 EngineCore 的增量结果汇总成用户可见的 `RequestOutput`。

对应 vLLM `v1/engine/output_processor.py`。它**不做**：重新决定接受/拒绝、操作 KV、在采样
内核里回调用户——那些分别属于 Scheduler 与执行侧。用户回调（将来的流式输出）落在这里。

它维护的是"用户侧已收到的累计结果"：

    request_id → RequestState(prompt_token_ids, token_ids=[], finished=False)

**只在把结束的那条 RequestOutput 交出去之后**才删掉状态，所以：

- 同一个 ID 在旧状态还没交付时不能复用（`add_request` 会报错）——这正是 195 §4 要求的
  "禁止在旧 ID 清理消息尚未消费时复用它"。vLLM 用代际编号解决，本关用"状态还在就拒绝"。
- 用户拿到的那份 `token_ids` 是**新 list**（快照），它继续拼接不会影响内部状态。
"""

from dataclasses import dataclass, field

from ..outputs import RequestOutput


@dataclass
class RequestState:
    """一条请求的用户侧状态。`token_ids` 是**只增不改**的累计输出。"""

    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int] = field(default_factory=list)
    finished: bool = False


@dataclass
class OutputProcessorResult:
    """一次 `process_outputs()` 的产物。"""

    request_outputs: list[RequestOutput] = field(default_factory=list)
    #: 需要 LLMEngine 回头去 abort 的请求（vLLM 用它处理 stop string 之类；本关恒空，留着接口）
    reqs_to_abort: list[str] = field(default_factory=list)

    @property
    def finished_request_ids(self) -> set[str]:
        return {out.request_id for out in self.request_outputs if out.finished}


class OutputProcessor:
    def __init__(self, tokenizer=None) -> None:
        # tokenizer 是 57B 的事（本关 no text）
        self.tokenizer = tokenizer
        self.request_states: dict[str, RequestState] = {}

    def add_request(self, request_id: str, prompt_token_ids: list[int]) -> None:
        if request_id in self.request_states:
            raise ValueError(
                f"请求 ID {request_id!r} 的用户侧状态还在（上一轮的结束结果还没交付）："
                f"不许复用同一个 ID")
        self.request_states[request_id] = RequestState(request_id=request_id,
                                                       prompt_token_ids=list(prompt_token_ids))

    def process_outputs(self, engine_core_outputs) -> OutputProcessorResult:
        """增量 → 累计。结束的那一条交出去之后，状态就删掉（ID 从此可以复用）。"""
        result = OutputProcessorResult()
        for core_output in engine_core_outputs.outputs:
            state = self.request_states.get(core_output.request_id)
            if state is None:
                # 结构化输出重试 / 已经被 abort 掉的请求，可能在这里出现：跳过。
                # vLLM 也有这条防御（它更细，区分了多种"不该有状态"的情形）。
                continue
            state.token_ids.extend(core_output.new_token_ids)
            finished = core_output.finished
            if finished:
                state.finished = True

            result.request_outputs.append(RequestOutput(
                request_id=state.request_id,
                prompt_token_ids=list(state.prompt_token_ids),
                token_ids=list(state.token_ids),          # 快照：用户改它不影响内部状态
                finished=finished,
                finish_reason=core_output.finish_reason,
                text=self._decode(state.token_ids) if self.tokenizer is not None else None,
            ))

            if finished:
                # 交付之后才删：在此之前重复 ID 会被 `add_request()` 拒绝
                del self.request_states[state.request_id]
        return result

    def abort_requests(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.request_states.pop(request_id, None)

    def _decode(self, token_ids: list[int]) -> str:
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(token_ids)
        return ""
