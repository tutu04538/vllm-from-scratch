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

from collections import deque

from ..outputs import EngineCoreOutputs
from . import EngineCoreRequest  # noqa: F401  （类型提示用；实际转换在 add_request）


class EngineCore:
    def __init__(self, vllm_config, executor) -> None:
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.model_executor = executor
        # 失败态（205 §5）：一旦某一轮在"执行/采样/提议"里抛过异常，引擎就**停摆**。
        # 半轮状态（镜像已更新、权威输出未提交）不可能靠重试自愈，所以下一轮必须在
        # **调度之前**拒绝，而不是返回空输出假装这一轮正常。
        self.failure: str | None = None
        # Scheduler 与 KVCacheManager 由执行侧交回容量之后一起创建：容量来自执行侧，
        # 不能先建调度器再问容量（初始化顺序见 195 §8）。
        from ..core.kv_cache_manager import KVCacheManager
        from ..core.sched.scheduler import Scheduler

        cache_config = self.model_executor.get_cache_config()
        self.kv_cache_manager = KVCacheManager(
            cache_config, max_model_len=vllm_config.model_config.max_model_len,
            # 69 关：走 CUDA Graph 时 0 号块必须留白（padding 行会 clamp 到它）。
            # 判据用**最终解析出来的模式**（`VllmConfig._resolve_cudagraph_config` 已经
            # 把 None/enforce_eager/设备这些因素算完了），不是用户原始输入。
            reserve_null_block=vllm_config.compilation_config.cudagraph_mode
            .has_full_cudagraphs())
        # 容量定了之后**立刻交给执行侧**：执行侧要按它分配物理缓存并绑定到 Attention 层
        # （195 §8 的第三步）。顺序不能反——先建调度器再分配缓存的话，第一轮就可能排出
        # 执行侧根本没有物理存储的块。
        self.model_executor.initialize_kv_cache(cache_config)
        # 68 关：结构化输出的管理器在**引擎级**（一张卡一份 grammar 编译环境），Scheduler 与
        # 执行侧都通过它拿东西：Scheduler 要"每行允许哪些 token"的掩码，执行侧负责把掩码
        # 打到 logits 上（上游同一条分工，见 068 §2）。
        from ..structured_output import StructuredOutputManager

        self.structured_output_manager = StructuredOutputManager(vllm_config)
        # 70 关：先定"这一轮走不走异步"（要问 executor 支不支持），再选调度器类。
        # 异步与同步的差别只有两个钩子（占位符怎么加、结果回来怎么结账），所以是**同一个
        # Scheduler 的子类**，不是两套调度逻辑（需求 070 §2：不复制整个 schedule）。
        from ..config import resolve_async_scheduling

        self.async_scheduling = resolve_async_scheduling(
            vllm_config, self.model_executor.supports_async_scheduling())
        scheduler_cls = Scheduler
        if self.async_scheduling:
            from ..core.sched.async_scheduler import AsyncScheduler

            scheduler_cls = AsyncScheduler
        self.scheduler = scheduler_cls(vllm_config.scheduler_config, self.kv_cache_manager,
                                       max_model_len=vllm_config.model_config.max_model_len,
                                       speculative_config=vllm_config.speculative_config,
                                       structured_output_manager=self.structured_output_manager)
        # 执行/采样 future 的队列：**有界**（上游 `batch_queue_size = 1 + pp_size`）。
        # 本仓库没有流水线并行，所以是 2：允许"排下一轮"与"上一轮还在跑"重叠一层。
        self.batch_queue_size = 2
        self.batch_queue = deque() if self.async_scheduling else None

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
        self._validate_sampling_params(sampling_params)
        room = max_model_len - prompt_len
        if sampling_params.max_tokens > room:
            sampling_params = dataclasses.replace(sampling_params, max_tokens=room)
            request = dataclasses.replace(request, sampling_params=sampling_params)
        # 块 hash 计算器属于控制面（要 block_size 与"是否开前缀缓存"），由 Scheduler 持有；
        # 请求一进门就挂上，保证 hash 链从第一个块开始就是完整的
        return Request.from_engine_core_request(request, block_hasher=self.scheduler.block_hasher)

    def _validate_sampling_params(self, sampling_params) -> None:
        """68 关的请求期校验（上游在 `Processor._validate_sampling_params` 的位置）。

        只做两件与**引擎配置**相关、逐请求看不出来的事：

        - `logprobs` 不能超过 `ModelConfig.max_logprobs`（静默截断会让用户以为拿到了全部）；
        - 结构化输出的规格要在**提交时**就校验/规范化（`choice` 会被改写成 EBNF；
          写错的 schema 在这里报错，而不是排到队之后才失败）。
        """
        if sampling_params is None:
            return
        num_logprobs = sampling_params.num_logprobs
        if num_logprobs is not None and num_logprobs != -1:
            model_config = self.vllm_config.model_config
            max_logprobs = model_config.max_logprobs
            if max_logprobs == -1:
                max_logprobs = model_config.get_vocab_size()
            if num_logprobs > max_logprobs:
                raise ValueError(
                    f"请求要 {num_logprobs} 个 logprobs，超过 max_logprobs={max_logprobs}"
                    f"（引擎配置）。上游同样在请求期拒绝：静默少给几个会让下游以为"
                    f"拿到的是完整的前 k 名")
        if sampling_params.structured_outputs is not None:
            from ..structured_output import validate_structured_output

            validate_structured_output(sampling_params,
                                       self.vllm_config.structured_outputs_config,
                                       tokenizer=None)

    def add_request(self, request) -> None:
        # 68 关：结构化输出的 grammar 在这里编译（上游也在 `EngineCore.add_request` 里调
        # `grammar_init`，不在 Scheduler 里）。**同步编译**，所以语法错的请求在提交处就失败，
        # 不会先排队再挂掉（异步编译属"本项目尚未接入"，见三态矩阵）。
        self.structured_output_manager.grammar_init(request)
        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        from ..request import RequestStatus

        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    def has_requests(self) -> bool:
        return self.scheduler.has_requests()

    # -------- 一轮 --------

    def step_with_batch_queue(self) -> tuple[EngineCoreOutputs, bool]:
        """异步调度的一轮（对应上游 `EngineCore.step_with_batch_queue()`，L624-738）。

        与同步 `step()` 的**唯一区别**是"谁等谁"：

            同步：排 → 跑 → 采 → **等结果** → 提交 → 再排下一轮（CPU 与 GPU 交替空转）
            异步：排 → 跑 → 采 → 入队 → **立刻去排下一轮**；队列满了才回头等最早那一轮

        于是"调度"（预算、KV 分配、占位记账）发生在上一轮结果还没回来的时候——这正是占位符
        存在的理由：排进去的宽度是**乐观的**，结果回来时再结账。

        `deferred_scheduler_output`（68 关结构化输出）：语法掩码要按"上一轮真实裁过的草稿"
        算，而那一刻草稿还没回来，所以这一轮的**采样**推迟到上一轮结果处理完之后（上游同一
        分支，连处理顺序都一致）。
        """
        try:
            return self._step_with_batch_queue()
        except Exception as exc:                     # noqa: BLE001 —— 半轮状态不可重试
            self.failure = f"{type(exc).__name__}: {exc}"
            raise

    def _step_with_batch_queue(self) -> tuple[EngineCoreOutputs, bool]:
        batch_queue = self.batch_queue
        assert batch_queue is not None
        assert len(batch_queue) < self.batch_queue_size, "队列不该超过容量"

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
            model_executed = scheduler_output.total_num_scheduled_tokens > 0
            if model_executed:
                if not scheduler_output.has_structured_output_requests:
                    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
                    future = self.model_executor.sample_tokens(grammar_output, non_block=True)
                else:
                    deferred_scheduler_output = scheduler_output
                    future = exec_future
            else:
                future = exec_future
            if deferred_scheduler_output is None:
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if len(batch_queue) < self.batch_queue_size and (
                        model_executed or self.scheduler.has_requests()):
                    # 队列没满：**不等结果**，交一个空输出回去让上层再调一次
                    return EngineCoreOutputs(), model_executed
        elif not batch_queue:
            return EngineCoreOutputs(), False

        # 队列满了（或没东西可排）：等最早那一轮，提交它的结果
        future, scheduler_output, exec_future = batch_queue.pop()
        model_output = future.result()
        if model_output is None:
            exec_future.result()
            raise RuntimeError("sample_tokens() 交回 None：说明 execute_model() 那一步失败了")
        # **交付边界**：异步句柄在这里才真的等拷贝完成（同步路径拿到的已经是 CPU 结果）
        if hasattr(model_output, "get_output"):
            model_output = model_output.get_output()
        engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

        if deferred_scheduler_output is not None:
            # 上一轮结果处理完了（草稿也裁过了）→ 现在可以算掩码并发起这一轮的采样
            grammar_output = self.scheduler.get_grammar_bitmask(deferred_scheduler_output)
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))
        return engine_core_outputs, model_executed

    def step(self) -> tuple[EngineCoreOutputs, bool]:
        """调度、执行、采样、更新。返回 (本轮输出, 是否真的跑了模型)。

        `total_num_scheduled_tokens == 0` 的轮（结束清理轮、或暂时没东西可排）**也会**调用
        `execute_model`：执行侧看到 0 token 就不碰 GPU、直接回一个空结果——"空轮不执行模型"
        这条约束落在执行侧（vLLM 的 runner 也是这么分的），Scheduler 不必知道。

        **失败即停摆**（205 §5）：只要有一轮抛过异常，这里就记下原因；之后每次 `step()`
        都在 `schedule()` **之前**直接拒绝——不推进请求、不改状态、也不返回空输出。
        执行侧的 `ModelRunner` 自己也会记 `failure`（它的 `_check_usable()` 是第二道门）。
        """
        self._check_alive()
        if not self.scheduler.has_requests():
            return EngineCoreOutputs(), False

        try:
            scheduler_output = self.scheduler.schedule()
            model_output = self.model_executor.execute_model(scheduler_output)
            if model_output is None:
                # 执行侧说"我先把状态存下了，你来采"——采样在这一步做
                # 68 关：**采样之前**先按语法算出"每个待定位置允许哪些 token"
                # （上游同序：`get_grammar_bitmask` → `sample_tokens(grammar_output)`）。
                # 掩码由执行侧打进 logits，`-1`（不约束）的行也在这里被填成全允许。
                grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
                model_output = self.model_executor.sample_tokens(grammar_output)
            outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        except Exception as exc:                     # noqa: BLE001 —— 半轮状态不可重试
            self.failure = f"{type(exc).__name__}: {exc}"
            raise
        return outputs, scheduler_output.total_num_scheduled_tokens > 0

    def _check_alive(self) -> None:
        """下一轮开始前先看失败标记：**在调度之前**拒绝，别让 Scheduler 的状态再往前走。"""
        if self.failure is not None:
            raise RuntimeError(
                f"EngineCore 已经失败，不再接受新的 step()：{self.failure}。"
                f"半轮状态（镜像已更新、权威输出未提交）不能靠继续调度自愈；"
                f"本关不做故障恢复，请重建引擎（197 §9 的失败策略）")

    def post_step(self, model_executed: bool) -> None:
        """执行之后的收尾：把执行侧提的草稿收进 Scheduler（供**下一轮**采用）。

        70 关（异步调度）：这一步**不做**——上游注释原话："when using async scheduling we
        can't get draft token ids in advance, so we update draft token ids in the worker
        process"。异步下 Scheduler 只按占位宽度排计划，真实草稿留在执行侧。
        """
        if self.async_scheduling:
            return
        if model_executed and self.speculative_config is not None:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            self.scheduler.update_draft_token_ids(draft_token_ids)

    def shutdown(self) -> None:
        self.model_executor.shutdown()
