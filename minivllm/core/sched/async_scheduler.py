"""异步调度器（对应 vLLM `v1/core/sched/async_scheduler.py`，L12-70）。

**它解决什么**：同步调度下 CPU 必须等这一轮的 GPU 结果拷回来，才知道"实际生成了什么"，
然后才敢排下一轮。异步调度让 CPU **不等**：调度完这一轮，立刻接着排下一轮。

代价是"排下一轮时还不知道上一轮的结果"，于是要引入**占位符**：

    num_output_placeholders   乐观预留的输出位置个数（最多 K+1 个）
    spec_token_ids = [-1]*K   乐观预留的草稿宽度（值还不知道，宽度先占上）

结果回来后再"结账"：按实际交付的长度减掉占位、按实际被拒数回退进度与占位。

**为什么只覆写两个钩子**（上游同款做法）：`schedule()` / `update_from_output()` 的主干与同步
完全一样，异步只改"调度之后记什么账"与"结果回来时怎么结账"这两处。需求 070 §2 明确要求
"不复制整个 schedule"——两份调度逻辑迟早会分叉，而分叉的症状是"某条路径下占位没结清"。

**它不做什么**：`num_stale_output_tokens` 的抵扣、`is_stale` 的保护、发布上界减去占位，
这些都已经在基类里按上游语义实现好了（`Scheduler.update_from_output` / `_publish_blocks`
/ `publish_bound`），异步侧只负责"加上占位"和"结账时用占位"——这样同步路径一行都不用改，
也不会有两套账。
"""

from ...request import Request
from ...core.sched.output import SchedulerOutput
from .scheduler import NUM_SAMPLED_TOKENS_PER_STEP, Scheduler


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 可复用的"占位草稿"列表：每轮按本轮要排的草稿数切片（上游同款，避免每轮新建）
        self._spec_token_placeholders: list[int] = [-1] * self.num_speculative_tokens

    # -------- 调度之后：加占位 --------

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """先让基类推进进度，再给**每条被调度的非 prefill 请求**记上占位（上游同序）。

        占位宽度 = 本步采样数（1） + 本轮采用的草稿数：因为这一轮最多会产出这么多新 token
        （草稿全被接受时就是 K 枚 + 1 个 bonus）。

        `spec_token_ids` 被设成**占位列表**（全是 -1）：下一轮 `schedule()` 靠它知道"要留多宽"，
        而**值**要到结果回来之后才有——执行侧在 GPU 上自己知道真实草稿（上游把这一步放在
        worker 里做，注释原话："when using async scheduling we can't get draft token ids in
        advance, so we update draft token ids in the worker process"）。

        中间 prefill 块跳过：它这一轮不产出 token，也不需要草稿宽度。
        """
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_spec_to_schedule = len(self._spec_token_placeholders)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                NUM_SAMPLED_TOKENS_PER_STEP + cur_num_spec_tokens)
            # 占位草稿：宽度按**下一轮**要排的草稿数给（本轮的草稿已经用掉了）
            # （`num_in_flight_tokens` 由基类在 `_update_after_schedule()` 里累加 ✓）
            request.spec_token_ids = self._spec_token_placeholders[:num_spec_to_schedule]

    # -------- 结果回来：结账 --------

    def _update_request_with_output(self, request: Request, new_token_ids: list[int],
                                    is_stale: bool = False) -> tuple[list[int], bool]:
        """逐 token 提交，然后**按实际交付长度**减占位（上游同名方法）。

        顺序不能反：`super()` 会把停下来的请求截断（`new_token_ids` 变短），占位必须按
        **截断之后**的长度减——多减的那部分会让占位变成负数（下游断言会报，但那已经是
        状态被写坏之后了）。

        `is_stale=True`（抢占前那一轮的结果）：只交付 token，**不动占位**——抢占时占位已经
        清零，再减就是负数（上游注释："a stale delivery must not decrement them"）。
        """
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids, is_stale=is_stale)
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            if request.num_output_placeholders < 0:
                raise RuntimeError(
                    f"{request.request_id!r} 的占位减成了负数：占位记账与实际交付不一致"
                    f"（上游同款断言；常见原因：同一轮的结果被结账两次，或占位加漏了）")
        return new_token_ids, stopped

    @staticmethod
    def publish_bound(request: Request) -> int:
        """可发布上界 = 已确认进度 − 未兑现的占位（上游把这条写在 `cache_blocks()` 调用处）。

        发布早了会让别的请求命中"预计会被接受、但还没真正算出来"的 KV——不报错、读到垃圾。
        """
        return request.num_computed_tokens - request.num_output_placeholders
