"""EngineCore 的客户端（对应 vLLM `v1/engine/core_client.py`）。

为什么要有这一层：`LLMEngine` 不该知道 EngineCore 是**同进程**还是**另一个进程**。它只调
`add_request` / `get_output` / `abort_requests` / `shutdown`；谁来跑这些调用由客户端决定。

真实 vLLM 有三种客户端（同进程、多进程 ZMQ、DP 分片），`EngineCoreClient` 是几十个方法的
ABC。本关只有一种实现，所以：

- `EngineCoreClient` 用 **Protocol**（小、只列本关会用到的四个方法），不写 ABC——
  为唯一的实现写抽象基类只是噪声（195 §2 明确允许）；
- `InprocClient` 只做两件事：把 `EngineCoreRequest` 预处理成 `Request` 再交给 EngineCore；
  以及 `get_output()` 里 **step + post_step** 两步（真实 InprocClient 就是主动 stepping，
  不要求后台忙轮询）。
"""

from typing import Protocol

from ..outputs import EngineCoreOutputs, EngineCoreRequest


class EngineCoreClient(Protocol):
    """本关用到的全部契约。加方法要同时想清楚：多进程实现能提供它吗？"""

    def add_request(self, request: EngineCoreRequest) -> None: ...

    def get_output(self) -> EngineCoreOutputs: ...

    def abort_requests(self, request_ids: list[str]) -> None: ...

    def shutdown(self) -> None: ...


class InprocClient:
    """同进程直连：不序列化、不起线程，调用即执行。"""

    def __init__(self, engine_core) -> None:
        self.engine_core = engine_core

    def add_request(self, request: EngineCoreRequest) -> None:
        # API 数据 → 内部 Request 的转换在 EngineCore 里（它拥有 Request 类型与块 hash 策略）
        engine_request = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(engine_request)

    def get_output(self) -> EngineCoreOutputs:
        # 70 关：异步调度时走"带 batch queue 的一轮"（它自己决定"排队"还是"等结果"），
        # 同步路径一字未改。两条路的 `post_step` 都要调：它负责把草稿收进 Scheduler。
        engine_core = self.engine_core
        if getattr(engine_core, "async_scheduling", False):
            outputs, model_executed = engine_core.step_with_batch_queue()
        else:
            outputs, model_executed = engine_core.step()
        engine_core.post_step(model_executed)
        return outputs

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self) -> None:
        self.engine_core.shutdown()
