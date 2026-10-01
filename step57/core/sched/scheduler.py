"""Scheduler：把"谁这一轮算几个 token"算出来，并把进度记在自己这边。

对应 vLLM `v1/core/sched/scheduler.py`。本关是**同步单卡子集**：没有 encoder/connector/指标，
但块分配、前缀命中、抢占恢复、priority 这条主线是与本机对齐的。

### 它拥有什么

`requests` / `waiting` / `running` / `kv_cache_manager` / `finished_req_ids`。**请求的进度
（`num_computed_tokens`）由它自己维护**——执行侧只收到一份旧进度快照，不回头改 Scheduler
的对象。

### 一轮的顺序（196 §5）

1. 先排 running：算 token 数 → `allocate_slots`；失败就**抢占**一个 victim（撤销它本轮的
   计划、退回预算、释放块、放回 waiting）再重试，直到成功或者连自己都被抢占；
2. **本轮发生过抢占就不接纳 waiting**（`if not preempted_reqs`）：否则刚释放的块立刻被新请求
   吃掉，被抢占的请求永远回不来；
3. 接纳 waiting：先查**前缀命中**，再按"命中 + 已有 + 新增"核算容量；
4. 打包快照（进度是**旧值**）；5. `_update_after_schedule()` 才推进进度。

### 抢占不是"重新创建一个请求"（196 §6）

    free 块 → status = PREEMPTED → num_computed_tokens = 0 → spec 清空
    → num_preemptions += 1 → 原对象放回 waiting 队首

prompt、已提交输出、优先级、采样配置、**块 hash 链**都保留。恢复时 `get_computed_blocks()`
可能命中已经留在池子里的完整块，所以不必从物理位置 0 全部重算。

### 前缀缓存的发布时机（本关的显式教学差异）

发布（把完整块登记进索引）发生在**结果处理之后**（`update_from_output` 里调
`kv_cache_manager.cache_blocks`），而不是 vLLM 那样在 `allocate_slots` 里顺手发布。理由是
"发布的块必须已经写完 KV"这条安全性要求最直白：本轮的 KV 在 forward 之后才成立。
代价是放弃了一些更早的命中机会——是性能差异，不是正确性差异（197 §4 要求写进差异账本）。
"""

from ...request import Request, RequestStatus
from ..kv_cache_utils import BlockHasher
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .request_queue import create_request_queue
from .utils import check_stop

# 一轮最多采样出几个 token（普通解码就是 1；57E 投机时按验证结果变）。
_NUM_SAMPLED_TOKENS_PER_STEP = 1

# 连续多少轮排不出任何 token 就认定卡住（教学保护，见模块头）
_MAX_NO_PROGRESS_STEPS = 2

# 最近的 trace 保留多少轮（够了；trace 是给人看的，不是日志系统）
_MAX_TRACE_STEPS = 200


class Scheduler:
    def __init__(self, scheduler_config, kv_cache_manager, max_model_len: int) -> None:
        self.max_num_seqs = scheduler_config.max_num_seqs
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        self.policy = scheduler_config.policy
        self.max_model_len = max_model_len
        self.kv_cache_manager = kv_cache_manager

        # 前缀缓存的 hash 计算器：**只在开了缓存时才有**。没有它 → `block_hashes` 为空 →
        # 命中查询与发布都是空操作（"关掉缓存就跑同一套代码的另一条分支"）。
        self.enable_prefix_caching = bool(kv_cache_manager.enable_caching)
        self.block_hasher = (BlockHasher(kv_cache_manager.block_size)
                             if self.enable_prefix_caching else None)

        self.requests: dict[str, Request] = {}
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []
        # 已经结束、但还没送到执行侧的请求 ID。**下一轮**的 SchedulerOutput 才带走它
        # （执行侧据此删掉缓存的请求状态），所以"最后一个请求结束"之后还需要一轮清理。
        self.finished_req_ids: set[str] = set()
        # 上一轮被调度的请求：不在里面的请求要带上完整 token 历史（执行侧可能没有它的状态）
        self.prev_step_scheduled_req_ids: set[str] = set()

        # 统计与 trace（测试与排查用；vLLM 放 metrics）
        self.num_steps = 0
        self.num_no_progress_steps = 0
        self.num_preemptions = 0
        self.trace: list[dict] = []

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
        # 没挂 hash 计算器的请求（如直接构造的 Request）在这里补上：hash 链必须从请求
        # 一出生就跟着 token 历史走，中途补算会漏掉已经确定的块
        if self.block_hasher is not None:
            request.attach_block_hasher(self.block_hasher)
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
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        req_to_new_blocks: dict[str, object] = {}
        num_scheduled_tokens: dict[str, int] = {}
        num_hit_tokens: dict[str, int] = {}
        preempted_reqs: list[Request] = []
        token_budget = self.max_num_batched_tokens

        # ---- 1) 先排 running（用下标遍历而不是 for：抢占要在循环里删元素）----
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            num_new_tokens = self._num_new_tokens(request, token_budget)
            if num_new_tokens == 0:
                # 没有新 token 可算（例如已到上下文上限）。**跳过它、继续看后面的**——
                # vLLM 在这里也是 continue，允许后面的请求先跑。
                req_index += 1
                continue

            # 分配失败 → 抢占 victim → 重试（直到成功，或者连自己都被抢占）
            new_blocks = None
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
                if new_blocks is not None:
                    break
                victim = self._pick_victim()
                if victim is None:
                    break
                victim_index = self.running.index(victim)
                if victim_index < req_index:
                    req_index -= 1
                del self.running[victim_index]
                # victim 可能**已经在本轮计划里**：撤销它的记录并把预算退回来，
                # 否则那些已经释放的块会被发到执行侧（196 §5 明确点名的坑）
                if victim in scheduled_running_reqs:
                    scheduled_running_reqs.remove(victim)
                    restored = num_scheduled_tokens.pop(victim.request_id)
                    token_budget += restored
                    req_to_new_blocks.pop(victim.request_id, None)
                self._preempt_request(victim)
                preempted_reqs.append(victim)
                if victim is request:
                    # 能抢的都抢完了，连它自己都下去了：本轮这条排不了
                    new_blocks = None
                    break
            if new_blocks is None:
                # 排不下：**停止本轮的 running 排序**（后面的请求下一轮再说）。
                # 注意这里不 break 出整个 schedule()，仍然要走打包与进度更新。
                break

            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # ---- 2) 再接纳 waiting：**本轮发生过抢占就不接纳**（196 §5.6）----
        while self.waiting and token_budget > 0 and not preempted_reqs:
            # 只数 running：本轮刚接纳的请求已经 append 进 running 了，
            # 再和 scheduled_new_reqs 相加会把同一条数两遍（那一版会让并发上限变成一半）
            if len(self.running) >= self.max_num_seqs:
                break
            request = self.waiting.peek_request()

            # 前缀命中：能白拿多少 token 的 KV（关缓存时恒为 0）
            computed_blocks, num_new_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
            start = request.num_computed_tokens + num_new_computed_tokens
            num_new_tokens = self._num_new_tokens(request, token_budget, start=start)
            if num_new_tokens == 0:
                break

            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=computed_blocks)
            if new_blocks is None:
                # 分配失败**什么都不改**：请求留在 waiting，下一轮再看（不许半个成功的块表）
                break

            was_preempted = request.status == RequestStatus.PREEMPTED
            self.waiting.pop_request()
            # 命中要在**打包快照之前**生效：本轮的起点就是"命中 + 已有进度"，
            # 执行侧据此跳过这段 token 的计算、直接读共享块的 KV
            request.num_computed_tokens = start
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            if was_preempted:
                scheduled_resumed_reqs.append(request)
            else:
                scheduled_new_reqs.append(request)
            if num_new_computed_tokens:
                num_hit_tokens[request.request_id] = num_new_computed_tokens
            # 首次接纳/恢复发的是**整张**块表（命中块 + 新块），之后每轮只发增量
            req_to_new_blocks[request.request_id] = self.kv_cache_manager.get_blocks(
                request.request_id)
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

        # ---- 3) 打包快照（进度是**旧值**）----
        new_reqs_data = [
            NewRequestData.from_request(request, req_to_new_blocks[request.request_id]
                                        .get_block_ids())
            for request in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs + scheduled_resumed_reqs, req_to_new_blocks,
            resumed_req_ids={request.request_id for request in scheduled_resumed_reqs})

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
        self._record_trace(scheduler_output, preempted_reqs, num_hit_tokens)

        # ---- 5) 计划成功之后，才推进自己的进度 ----
        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _num_new_tokens(self, request: Request, token_budget: int,
                        start: int | None = None) -> int:
        """统一预算公式：差多少、预算剩多少、上下文还装得下多少，三者取小。

        `start` 是本轮的起点（= 已算进度 + 前缀命中），默认就是 `num_computed_tokens`。
        """
        if start is None:
            start = request.num_computed_tokens
        num_new_tokens = request.num_tokens_with_spec - start
        num_new_tokens = min(num_new_tokens, token_budget)
        # 给本轮采样出来的 token 留位置：算到 start + n 之后还要能放下 1 个新 token
        num_new_tokens = min(num_new_tokens,
                             self.max_model_len - start - _NUM_SAMPLED_TOKENS_PER_STEP)
        return max(num_new_tokens, 0)

    def _make_cached_request_data(self, running_reqs: list[Request],
                                  req_to_new_blocks: dict, resumed_req_ids: set) -> CachedRequestData:
        req_ids, new_block_ids, num_computed_tokens, num_output_tokens = [], [], [], []
        all_token_ids: dict[str, list[int]] = {}
        for request in running_reqs:
            req_id = request.request_id
            req_ids.append(req_id)
            # 新 list：快照与 Scheduler 手里的块表不共享可变对象
            new_block_ids.append(req_to_new_blocks[req_id].get_block_ids(allow_none=True))
            # 注意：resumed 请求的"新增块"是**整张表**（见 sched/output.py 的说明），
            # 执行侧靠 resumed_req_ids 区分"替换"与"追加"两种语义
            num_computed_tokens.append(request.num_computed_tokens)     # 旧值
            num_output_tokens.append(request.num_output_tokens)
            if req_id not in self.prev_step_scheduled_req_ids:
                # 上一轮没被调度过（新接纳、或刚从抢占恢复）：执行侧可能没有它的历史，
                # 带上完整 token 列表。**不是每轮都复制全部历史**。
                all_token_ids[req_id] = list(request.all_token_ids)
        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=set(resumed_req_ids),
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
            all_token_ids=all_token_ids,
        )

    # -------- 抢占 --------

    def _pick_victim(self) -> Request | None:
        """挑一个牺牲者。**FCFS 取 running 尾部，priority 取优先级最低的那条**（196 §5）。

        数值大的 priority 优先级低、同优先级里到达晚的先让路（`max` 的键就是
        `(priority, arrival_time)`）。本关不做"新来的高优先级请求一定立刻赶走正在运行的请求"
        ——它只在**本轮分配失败**时才触发。
        """
        if not self.running:
            return None
        if self.policy == "priority":
            return max(self.running, key=lambda request: (request.priority, request.arrival_time))
        return self.running[-1]

    def _preempt_request(self, request: Request) -> None:
        """抢占：释放块、退回 waiting 队首、保留历史（196 §6）。

        **调用方负责把它从 running 里摘掉**（vLLM 也是这条约定），避免两边都删。
        """
        self.kv_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.spec_token_ids = []
        request.num_preemptions += 1
        self.num_preemptions += 1
        # 放到 waiting **队首**：它已经有一条历史了，应该比新来的先得到机会
        self.waiting.prepend_request(request)

    # -------- 空转保护 --------

    def _check_no_progress(self, scheduler_output: SchedulerOutput) -> None:
        """连续两轮一个 token 都排不出、却还有未完成请求 → 明确报错，不无限空转。

        vLLM 没有这条；这是本关的**教学保护**（196 §9.7：失败策略要可查，不能空转）。
        结束清理轮不算：那时 `has_unfinished_requests()` 已经是 False。

        抢占之后不该再走到这里——`_check_no_progress` 的报错信息里会带上 running 的进度，
        因为"排不出"在 57C 之后通常意味着"容量真的不够（比如 unary 请求就比池子大）"。
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
                "  常见原因：max_num_batched_tokens 不够、请求已到上下文上限、"
                "或单条请求需要的块比整个池子还大（抢占也救不了）。")

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """调度成功之后才推进进度（顺序不能反：快照里必须是旧值）。"""
        for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            # 派生判断：还没算到已有历史的末尾（中间 prefill 块）。57A 没有代码依赖它，
            # 但它是"这轮该不该产出 token"的权威口径，执行侧按同一口径判 ready。
            request.is_prefill_chunk = request.num_computed_tokens < request.num_tokens
        self.prev_step_scheduled_req_ids = set(scheduler_output.num_scheduled_tokens)
        # 重新绑定（不是 clear）：上面那个快照还引用着旧集合，它必须保持"本轮要清理的 ID"
        self.finished_req_ids = set()

    # -------- 结果回来之后 --------

    def update_from_output(self, scheduler_output: SchedulerOutput,
                           model_runner_output) -> object:
        """按 **request_id** 找结果、逐 token 提交、判停、发布缓存、收尾。"""
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

            # 发布缓存：此刻本轮的 KV 已经写完（forward 在前），进度也是校正过的值。
            # 放在"释放块之前"——释放之后块就进空闲队列了，但内容还在，只是我们要在
            # 还持有它的时候把 hash 登记好（197 §4）
            self._publish_blocks(request)

            should_emit = bool(new_token_ids) or stopped
            if not should_emit:
                # 中间 prefill 块：这一轮不产出 token，也不该给用户发消息
                continue

            finish_reason = None
            if stopped:
                finish_reason = request.get_finished_reason()
                # 上面刚发布过缓存，这里不再发布（publish=False）
                self._free_request(request, publish=False)
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

    def _publish_blocks(self, request: Request) -> None:
        """把这条请求已经确定的完整块登记进前缀缓存（关缓存时是空操作）。"""
        if self.enable_prefix_caching:
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

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
            self._free_request(request, publish=False)
            finished.append(request)
        return finished

    def _free_request(self, request: Request, publish: bool = True) -> None:
        """结束一条请求：发布缓存、还块、登记"要通知执行侧"、从活动索引里摘掉。

        `publish=False` 用于 **abort**：被取消的请求不该把它那半截历史发布出去——
        它可能是"用户不想要了"的中途状态，而且 abort 时本轮的 KV 未必写完。
        """
        if publish:
            self._publish_blocks(request)
        self.kv_cache_manager.free(request)
        self.finished_req_ids.add(request.request_id)
        self.requests.pop(request.request_id, None)

    # -------- trace --------

    def _record_trace(self, scheduler_output: SchedulerOutput, preempted_reqs: list[Request],
                      num_hit_tokens: dict[str, int]) -> None:
        """记一轮的调度决策。**给人看的**：抢占/命中/块占用三件事要一眼看出来。"""
        self.trace.append({
            "step": self.num_steps,
            "scheduled": dict(scheduler_output.num_scheduled_tokens),
            "hits": dict(num_hit_tokens),
            "preempted": [request.request_id for request in preempted_reqs],
            "running": [request.request_id for request in self.running],
            # 用 request_ids()：priority 队列是堆，不能直接迭代（而且堆里有懒惰删除的旧条目）
            "waiting": self.waiting.request_ids(),
            **self.kv_cache_manager.block_stats(),
        })
        if len(self.trace) > _MAX_TRACE_STEPS:
            del self.trace[0]

    def format_trace(self) -> str:
        """把 trace 排成一张表（`step57.py --scheduler-trace` 直接打印它）。"""
        lines = [f"{'step':>4} {'scheduled':<28} {'hits':<14} {'preempted':<12} "
                 f"{'running':<18} {'waiting':<14} {'free':>5} {'cached':>6}"]
        for record in self.trace:
            lines.append(
                f"{record['step']:>4} {str(record['scheduled']):<28} "
                f"{str(record['hits'] or ''):<14} {str(record['preempted'] or ''):<12} "
                f"{str(record['running']):<18} {str(record['waiting']):<14} "
                f"{record['free']:>5} {record['cached']:>6}")
        return "\n".join(lines)
