"""EngineCore：一轮的编排者（对应 vLLM `v1/engine/core.py::EngineCore`）。

它只做四件事：**调度 → 执行 → 采样 → 用结果更新调度器**。不遍历层、不写 KV、不拼 GPU 输入。

    调度：scheduler.schedule() -> SchedulerOutput        （数据包，不含活对象）
    执行：executor.execute_model(packet) -> ModelRunnerOutput | None
    采样：executor.sample_tokens(...) -> ModelRunnerOutput   （执行返回 None 时）
    更新：scheduler.update_from_output(packet, output) -> EngineCoreOutputs

**执行与采样分开**是刻意的：执行侧把 logits 等临时状态存在自己那边、先返回 None，再由
`sample_tokens()` 消费。这样采样（含将来的结构化输出、投机验证）就不必挤进模型前向里。
真实 vLLM 的 `execute_model(non_block=True)` 返回 Future；本关同步直调，**不造一个没有并发
的线程来假装异步**。

`post_step()` 只处理投机草稿（57E）。本关 `speculative_config is None`，所以它是空操作——
但接口留着，"草稿在一步之后才取回来"这条时序差异是真实存在的。
"""

from ..outputs import EngineCoreOutputs
from . import EngineCoreRequest  # noqa: F401  （类型提示用；实际转换在 add_request）


class EngineCore:
    def __init__(self, vllm_config, executor) -> None:
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.model_executor = executor
        # Scheduler 与 KVCacheManager 由执行侧交回容量之后一起创建：容量来自执行侧，
        # 不能先建调度器再问容量（初始化顺序见 195 §8）。
        from ..core.kv_cache_manager import KVCacheManager
        from ..core.sched.scheduler import Scheduler

        cache_config = self.model_executor.get_cache_config()
        self.kv_cache_manager = KVCacheManager(cache_config)
        # 容量定了之后**立刻交给执行侧**：执行侧要按它分配物理缓存并绑定到 Attention 层
        # （195 §8 的第三步）。顺序不能反——先建调度器再分配缓存的话，第一轮就可能排出
        # 执行侧根本没有物理存储的块。
        self.model_executor.initialize_kv_cache(cache_config)
        self.scheduler = Scheduler(vllm_config.scheduler_config, self.kv_cache_manager,
                                   max_model_len=vllm_config.model_config.max_model_len)

    # -------- 请求 --------

    def preprocess_add_request(self, request: EngineCoreRequest):
        """`EngineCoreRequest`（API 数据）→ `Request`（内部状态）。列表在这里复制一层。

        顺带做**输入边界检查**（对应 vLLM `v1/engine/processor.py` 的位置：校验发生在
        "外部数据变成内部请求"这一步，而不是散在调度器里）：

        - prompt 为空、或长到连一个 token 都生成不出来（`len(prompt) >= max_model_len`）→ 拒绝。
          不拦的话，Scheduler 只会表现为"连续两轮排不出 token"，报错信息指向调度器，而真正的问题
          在请求本身（57A 的差异账本里记过这个缺口）。
        - `max_tokens` 超出剩余上下文 → **截到装得下的量**（vLLM 同样这么做），不是拒绝：
          "生成 100 个"但只剩 20 个位置时，合理的语义是尽力生成。
        """
        import dataclasses

        from ..request import Request

        max_model_len = self.vllm_config.model_config.max_model_len
        prompt_len = len(request.prompt_token_ids)
        if prompt_len == 0:
            raise ValueError(f"{request.request_id!r} 的 prompt 是空的：至少要有一个 token")
        if prompt_len >= max_model_len:
            raise ValueError(
                f"{request.request_id!r} 的 prompt 有 {prompt_len} 个 token，"
                f"而 max_model_len={max_model_len}：连一个 token 都生成不出来。"
                f"（在入口拒绝，否则只会变成调度器的「排不出 token」空转报错，"
                f"看不出真正的问题在请求本身）")
        # 复制一份而不是原地改：`sampling_params` 是调用方的对象（还会被一并打包进
        # NewRequestData 发给执行侧），就地改会让"谁改了我的配置"说不清
        sampling_params = request.sampling_params
        room = max_model_len - prompt_len
        if sampling_params.max_tokens > room:
            sampling_params = dataclasses.replace(sampling_params, max_tokens=room)
            request = dataclasses.replace(request, sampling_params=sampling_params)
        return Request.from_engine_core_request(request)

    def add_request(self, request) -> None:
        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        from ..request import RequestStatus

        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    def has_requests(self) -> bool:
        return self.scheduler.has_requests()

    # -------- 一轮 --------

    def step(self) -> tuple[EngineCoreOutputs, bool]:
        """调度、执行、采样、更新。返回 (本轮输出, 是否真的跑了模型)。

        `total_num_scheduled_tokens == 0` 的轮（结束清理轮、或暂时没东西可排）**也会**调用
        `execute_model`：执行侧看到 0 token 就不碰 GPU、直接回一个空结果——"空轮不执行模型"
        这条约束落在执行侧（vLLM 的 runner 也是这么分的），Scheduler 不必知道。
        """
        if not self.scheduler.has_requests():
            return EngineCoreOutputs(), False

        scheduler_output = self.scheduler.schedule()
        model_output = self.model_executor.execute_model(scheduler_output)
        if model_output is None:
            # 执行侧说"我先把状态存下了，你来采"——采样在这一步做
            model_output = self.model_executor.sample_tokens(grammar_output=None)
        outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        return outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:
        """执行之后的收尾。57A 无投机，所以是空操作（接口按 195 §7 保留）。"""
        if model_executed and self.speculative_config is not None:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            self.scheduler.update_draft_token_ids(draft_token_ids)

    def shutdown(self) -> None:
        self.model_executor.shutdown()
