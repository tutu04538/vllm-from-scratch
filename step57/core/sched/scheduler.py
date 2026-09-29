"""Scheduler：把"谁这一轮算几个 token"算出来，并把进度记在自己这边。

对应 vLLM `v1/core/sched/scheduler.py`。本关只留**同步单卡子集**：没有抢占、没有投机、
没有 prefix 命中、没有 encoder、没有 connector、没有统计上报（各自属于 57C/57E）。

### 它拥有什么

`requests` / `waiting` / `running` / `kv_cache_manager` / `finished_req_ids`。**请求的进度
（`num_computed_tokens`）由它自己维护**——执行侧只收到一份旧进度快照，不回头改 Scheduler
的对象。这是本关最关键的一条：step56 让模型写 `seq.cache.length`、Scheduler 再去读，判断
"下一轮算谁"依赖了模型改同一个对象。

### 统一预算

没有"prefill 阶段"和"decode 阶段"，只有一条公式：

    num_new_tokens = min(num_tokens_with_spec - num_computed_tokens, token_budget, ...)

它自然解释 chunked prefill（prompt 长 → 只算一个 chunk）、普通 decode（差 1 个待定 token）、
抢占恢复（进度退回、要重算确定历史）、投机（确定历史末尾再接草稿）。

### 57A 与 vLLM 的差异（差异账本）

| 差异 | 原因 |
|---|---|
| **没有抢占** | 57C 接；本轮 `allocate_slots()` 失败就停止本轮排序（不挑 victim） |
| 没有 `skipped_waiting` 队列与阻塞状态 | 那些服务于 WAITING_FOR_REMOTE_KVS / streaming（本关无） |
| `_handle_stopped_request` 内联为"必定结束" | 它只为 streaming 会话服务 |
| **多了"连续空转报错"** | 教学保护：连续两轮一个 token 都排不出且仍有未完成请求 → 明确报错，不无限空转（196 §9.7 要求失败策略可查） |
| 没有 `preempted_req_ids` / `num_invalid_spec_tokens` 等字段 | V2/投机专用，本关沿 V1 且不投机 |
"""

from ...request import Request, RequestStatus
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .request_queue import create_request_queue
from .utils import check_stop

# 一轮最多采样出几个 token（普通解码就是 1；57E 投机时按验证结果变）。
# 它参与"给采样结果留位置"的上下文裁剪，所以放在这里显式命名。
_NUM_SAMPLED_TOKENS_PER_STEP = 1

# 连续多少轮排不出任何 token 就认定卡住（教学保护，见模块头）
_MAX_NO_PROGRESS_STEPS = 2


class Scheduler:
    def __init__(self, scheduler_config, kv_cache_manager, max_model_len: int) -> None:
        self.max_num_seqs = scheduler_config.max_num_seqs
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        self.policy = scheduler_config.policy
        self.max_model_len = max_model_len
        self.kv_cache_manager = kv_cache_manager

        self.requests: dict[str, Request] = {}
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []
        # 已经结束、但还没送到执行侧的请求 ID。**下一轮**的 SchedulerOutput 才带走它
        # （执行侧据此删掉缓存的请求状态），所以"最后一个请求结束"之后还需要一轮清理。
        self.finished_req_ids: set[str] = set()
        # 上一轮被调度的请求：不在里面的请求要带上完整 token 历史（执行侧可能没有它的状态）
        self.prev_step_scheduled_req_ids: set[str] = set()

        # 统计（测试与排查用；vLLM 放 metrics）
        self.num_steps = 0
        self.num_no_progress_steps = 0

    # -------- 请求进入/离开 --------

    def add_request(self, request: Request) -> None:
        """只做入队与登记。ID 冲突立即报错，两种情况都算冲突：

        - 活动请求里已经有它；
        - **它刚结束、但"可以清理了"的消息还没送到执行侧**（还在 `finished_req_ids` 里）。

        第二条为什么必要：下一轮的包里会**同时**出现"要清理 `r1`"和"新来的 `r1`"，执行侧
        先处理哪个都会把状态搞乱。vLLM 用代际编号（generation）解决；本关的选择是**直接
        禁止复用**——等清理消息送出去（一轮之后）再复用就可以。"""
        if request.request_id in self.requests:
            raise ValueError(
                f"活动请求里已经有 {request.request_id!r}：同一时刻不允许两个活动对象用同一个 ID")
        if request.request_id in self.finished_req_ids:
            raise ValueError(
                f"{request.request_id!r} 刚结束，但它的清理消息还没送到执行侧；"
                f"此时复用同一个 ID 会让下一轮的包同时包含「清理 r1」与「新增 r1」，"
                f"执行侧的请求状态必然错乱。等下一轮（清理消息送出去）之后再复用。")
        request.status = RequestStatus.WAITING
        self.waiting.add_request(request)
        self.requests[request.request_id] = request

    def has_unfinished_requests(self) -> bool:
        return bool(self.running) or bool(self.waiting)

    def has_finished_requests(self) -> bool:
        """有请求结束、但执行侧还没收到那条清理消息。"""
        return bool(self.finished_req_ids)

    def has_requests(self) -> bool:
        """引擎是否还该继续 step：还有活干，或者还有清理消息没送出去。"""
        return self.has_unfinished_requests() or self.has_finished_requests()

    def get_request_counts(self) -> tuple[int, int]:
        return len(self.running), len(self.waiting)

    # -------- 一轮调度 --------

    def schedule(self) -> SchedulerOutput:
        self.num_steps += 1
        scheduled_new_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        req_to_new_blocks: dict[str, object] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_batched_tokens

        # ---- 1) 先排 running（用下标遍历而不是 for：57C 的抢占要在循环里删元素）----
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            num_new_tokens = self._num_new_tokens(request, token_budget)
            if num_new_tokens == 0:
                # 没有新 token 可算（例如已到上下文上限）。**跳过它、继续看后面的**——
                # vLLM 在这里也是 continue，允许后面的请求先跑。
                req_index += 1
                continue

            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
            if new_blocks is None:
                # 57A：容量不够时不抢占，本轮不再往下排（57C 在这里挑 victim 并重试）
                break

            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # ---- 2) 再接纳 waiting（57A 没有抢占，所以不判"本轮是否发生过抢占"）----
        while self.waiting and token_budget > 0:
            # 只数 running：本轮刚接纳的请求已经 append 进 running 了，
            # 再和 scheduled_new_reqs 相加会把同一条数两遍（那一版会让并发上限变成一半）
            if len(self.running) >= self.max_num_seqs:
                break
            request = self.waiting.peek_request()
            num_new_tokens = self._num_new_tokens(request, token_budget)
            if num_new_tokens == 0:
                break
            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
            if new_blocks is None:
                # 分配失败**什么都不改**：请求留在 waiting，下一轮再看（不许半个成功的块表）
                break
            self.waiting.pop_request()
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            scheduled_new_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

        # ---- 3) 打包快照（进度是**旧值**）----
        new_reqs_data = [
            NewRequestData.from_request(request, req_to_new_blocks[request.request_id]
                                        .get_block_ids())
            for request in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(scheduled_running_reqs, req_to_new_blocks)

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},          # 57E 才填
            # 注意是**引用**当前的集合，下面 `_update_after_schedule()` 会把它绑定到新对象上；
            # 清空（而不是重新绑定）会让这里已经发出去的快照跟着变空。
            finished_req_ids=self.finished_req_ids,
        )

        # ---- 4) 空转保护（教学扩展）----
        self._check_no_progress(scheduler_output)

        # ---- 5) 计划成功之后，才推进自己的进度 ----
        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _num_new_tokens(self, request: Request, token_budget: int) -> int:
        """统一预算公式：差多少、预算剩多少、上下文还装得下多少，三者取小。"""
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)
        # 给本轮采样出来的 token 留位置：算到 num_computed + n 之后还要能放下 1 个新 token
        num_new_tokens = min(num_new_tokens,
                             self.max_model_len - request.num_computed_tokens
                             - _NUM_SAMPLED_TOKENS_PER_STEP)
        return max(num_new_tokens, 0)

    def _make_cached_request_data(self, running_reqs: list[Request],
                                  req_to_new_blocks: dict) -> CachedRequestData:
        req_ids, new_block_ids, num_computed_tokens, num_output_tokens = [], [], [], []
        all_token_ids: dict[str, list[int]] = {}
        for request in running_reqs:
            req_id = request.request_id
            req_ids.append(req_id)
            # 新 list：快照与 Scheduler 手里的块表不共享可变对象
            new_block_ids.append(req_to_new_blocks[req_id].get_block_ids(allow_none=True))
            num_computed_tokens.append(request.num_computed_tokens)     # 旧值
            num_output_tokens.append(request.num_output_tokens)
            if req_id not in self.prev_step_scheduled_req_ids:
                # 上一轮没被调度过（新接纳、或刚从抢占恢复）：执行侧可能没有它的历史，
                # 带上完整 token 列表。**不是每轮都复制全部历史**。
                all_token_ids[req_id] = list(request.all_token_ids)
        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=set(),        # 57A 没有抢占，恢复属 57C
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
            all_token_ids=all_token_ids,
        )

    def _check_no_progress(self, scheduler_output: SchedulerOutput) -> None:
        """连续两轮一个 token 都排不出、却还有未完成请求 → 明确报错，不无限空转。

        vLLM 没有这条；这是本关的**教学保护**（196 §9.7：失败策略要可查，不能空转）。
        结束清理轮不算：那时 `has_unfinished_requests()` 已经是 False。
        """
        if scheduler_output.total_num_scheduled_tokens > 0:
            self.num_no_progress_steps = 0
            return
        if not self.has_unfinished_requests():
            self.num_no_progress_steps = 0
            return
        self.num_no_progress_steps += 1
        if self.num_no_progress_steps >= _MAX_NO_PROGRESS_STEPS:
            raise RuntimeError(
                "调度连续排不出任何 token，且仍有未完成请求：\n"
                f"  running={[(r.request_id, r.num_tokens, r.num_computed_tokens) for r in self.running]}\n"
                f"  waiting={[r.request_id for r in self.waiting]}\n"
                f"  空闲块={self.kv_cache_manager.num_free_blocks()} / "
                f"总块={self.kv_cache_manager.num_gpu_blocks}\n"
                "  常见原因：max_num_batched_tokens 或块容量不够、请求已到上下文上限、"
                "或抢占未实现（57C）导致容量被占住。")

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """调度成功之后才推进进度（顺序不能反：快照里必须是旧值）。"""
        for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            # 派生判断：还没算到已有历史的末尾（中间 prefill 块）。57A 没有代码依赖它，
            # 但它是"这轮该不该产出 token"的权威口径，留着给 57B 的 Runner 对齐。
            request.is_prefill_chunk = request.num_computed_tokens < request.num_tokens
        self.prev_step_scheduled_req_ids = set(scheduler_output.num_scheduled_tokens)
        # 重新绑定（不是 clear）：上面那个快照还引用着旧集合，它必须保持"本轮要清理的 ID"
        self.finished_req_ids = set()

    # -------- 结果回来之后 --------

    def update_from_output(self, scheduler_output: SchedulerOutput,
                           model_runner_output) -> object:
        """按 **request_id** 找结果、逐 token 提交、判停、收尾。"""
        from ...outputs import EngineCoreOutput, EngineCoreOutputs

        outputs: list[EngineCoreOutput] = []
        stopped_running: list[Request] = []
        finished_now: set[str] = set()

        # 按 Scheduler 自己的 num_scheduled_tokens 遍历（顺序稳定），再用 req_id_to_index
        # 去结果里取——**不能**按 Runner 的行顺序 zip（Runner 允许重排）。
        for req_id, _num_scheduled in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # 执行期间被 abort 掉了：跳过（它的收尾已经在 abort 路径里做过）
                continue
            req_index = model_runner_output.req_id_to_index[req_id]
            new_token_ids = list(model_runner_output.sampled_token_ids[req_index])

            stopped = False
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)

            should_emit = bool(new_token_ids) or stopped
            if not should_emit:
                # 中间 prefill 块：这一轮不产出 token，也不该给用户发消息
                continue

            finish_reason = None
            if stopped:
                finish_reason = request.get_finished_reason()
                self._free_request(request)
                stopped_running.append(request)
                finished_now.add(req_id)

            outputs.append(EngineCoreOutput(
                request_id=req_id,
                new_token_ids=new_token_ids,
                finish_reason=finish_reason,
                stop_reason=request.stop_reason,
            ))

        if stopped_running:
            stopped_ids = {request.request_id for request in stopped_running}
            self.running = [r for r in self.running if r.request_id not in stopped_ids]

        return EngineCoreOutputs(outputs=outputs, finished_requests=finished_now)

    def _update_request_with_output(self, request: Request,
                                    new_token_ids: list[int]) -> tuple[list[int], bool]:
        """逐 token 提交 + 判停。停下就**截断本轮剩余候选**（后面的 token 不提交）。"""
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]
                break
        return new_token_ids, stopped

    # -------- 结束与清理 --------

    def finish_requests(self, request_ids, finished_status: RequestStatus) -> list[Request]:
        """外部（abort / shutdown）要求结束。已经结束或不存在的不算。"""
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        finished = []
        for req_id in list(request_ids):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            if request.status == RequestStatus.RUNNING:
                self.running = [r for r in self.running if r.request_id != req_id]
            else:
                self.waiting.remove_request(request)
            request.status = finished_status
            self._free_request(request)
            finished.append(request)
        return finished

    def _free_request(self, request: Request) -> None:
        """结束一条请求：还块、登记"要通知执行侧"、从活动索引里摘掉。"""
        self.kv_cache_manager.free(request)
        self.finished_req_ids.add(request.request_id)
        self.requests.pop(request.request_id, None)
