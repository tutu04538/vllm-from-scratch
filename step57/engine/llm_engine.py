"""LLMEngine：用户侧入口（对应 vLLM `v1/engine/llm_engine.py`）。

职责边界（195 §2）：

- **持有**：`engine_core`（一个客户端）、`output_processor`；
- **做**：`add_request` / `step` / `abort_request` / `has_unfinished_requests`；
- **不做**：分配块、拼 GPU 输入、决定接受/拒绝。

`step()` 就三步：向客户端要一轮结果 → 交给 `OutputProcessor` 汇总成用户可见输出 →
（本关恒空的）abort 列表处理。真实 vLLM 还在这里记账、上报指标、处理 stop string。

57A 的输入只接受 **token IDs**：字符串编码属于 57B 的输入边界（tokenizer / 多模态渲染）。
"""

import time

from ..outputs import EngineCoreRequest
from .core import EngineCore
from .core_client import InprocClient
from .output_processor import OutputProcessor


class LLMEngine:
    def __init__(self, vllm_config, executor, tokenizer=None) -> None:
        self.vllm_config = vllm_config
        self.engine_core = InprocClient(EngineCore(vllm_config, executor))
        self.output_processor = OutputProcessor(tokenizer=tokenizer)

    # -------- 请求 --------

    def add_request(self, request_id: str, prompt_token_ids: list[int], sampling_params,
                    arrival_time: float | None = None, priority: int = 0) -> str:
        """提交一条请求。**在这里就做重复 ID 校验**（用户侧状态还在就直接报错）。"""
        if not isinstance(request_id, str):
            raise TypeError(f"request_id 必须是字符串，收到 {type(request_id)}")

        # 先占住用户侧状态：重复 ID 在这一步就会报错，不会走到 EngineCore
        self.output_processor.add_request(request_id, prompt_token_ids)

        engine_core_request = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),      # 复制：用户之后改自己那份无影响
            sampling_params=sampling_params,
            arrival_time=arrival_time if arrival_time is not None else time.time(),
            priority=priority,
        )
        try:
            self.engine_core.add_request(engine_core_request)
        except Exception:
            # 调度器那边拒绝了（ID 还在等清理、或已有活动请求）：**回滚用户侧登记**，
            # 否则会留下一个"有输出状态、但引擎里没有对应请求"的孤儿，之后这个 ID 再也用不了。
            self.output_processor.abort_requests([request_id])
            raise
        return request_id

    def abort_request(self, request_ids: list[str]) -> None:
        self.output_processor.abort_requests(request_ids)
        self.engine_core.abort_requests(request_ids)

    def has_unfinished_requests(self) -> bool:
        return self.engine_core.engine_core.has_requests()

    def get_num_unfinished_requests(self) -> int:
        running, waiting = self.engine_core.engine_core.scheduler.get_request_counts()
        return running + waiting

    # -------- 一轮 --------

    def step(self):
        """跑一轮，返回本轮的用户可见输出（可能是空的：中间 prefill 块不产出）。"""
        outputs = self.engine_core.get_output()
        processed = self.output_processor.process_outputs(outputs)
        if processed.reqs_to_abort:
            self.abort_request(processed.reqs_to_abort)
        return processed.request_outputs

    def shutdown(self) -> None:
        self.engine_core.shutdown()
