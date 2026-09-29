"""`FakeRunner`：**只给测试用**的脚本化 Runner（生产路径不 import 这个包）。

它存在的理由不是"省事"，而是**把协议当成唯一的输入**：一个不读 Scheduler 的执行侧能不能
只靠 `SchedulerOutput` 干活？能，就说明边界是干净的。它做的三件事正是真实 Runner 在 57B/57D
必须做的：

1. 从 `NewRequestData` / `CachedRequestData` 维护**自己那份**请求镜像（prompt、进度、块表）；
2. 按 `finished_req_ids` 删掉镜像（结束清理）；
3. `execute_model()` 存下计划返回 `None`，`sample_tokens()` 才产出 `ModelRunnerOutput`。

另外它**主动检查协议违规**：包里有活对象、闭包回调，或者出现内部 `Request`，立刻记下来并
报错——这条约束靠"看一眼代码"守不住，要靠执行侧真的拒收。

关于"产出什么 token"：本关**要求测试显式给脚本**（`tokens={"r1": [11, 12]}`），脚本用完就报错。
不提供"随便编一个"的兜底——那样测试会依赖一个看不见的伪随机序列，断言就不可读了。
"""

from dataclasses import dataclass, field

from ..outputs import ModelRunnerOutput
from ..request import Request


@dataclass
class RunnerRequestState:
    """执行侧自己的请求镜像。它与 Scheduler 的 Request 是**两份状态**，靠协议数据对齐。"""

    prompt_token_ids: list[int]
    num_computed_tokens: int = 0
    emitted: list[int] = field(default_factory=list)      # 已产出的 token（可能被校正）
    block_ids: tuple[list[int], ...] = ()

    @property
    def all_token_ids(self) -> list[int]:
        return list(self.prompt_token_ids) + list(self.emitted)


class FakeRunner:
    def __init__(self, tokens: dict[str, list[int]] | None = None,
                 reverse_rows: bool = False) -> None:
        self.tokens = {req_id: list(script) for req_id, script in (tokens or {}).items()}
        self.reverse_rows = reverse_rows
        self.req_states: dict[str, RunnerRequestState] = {}
        self.pending = None
        # 观测点（测试断言用）
        self.num_forward_calls = 0          # 真的"跑模型"的次数（空轮不算）
        self.num_forward_tokens = 0
        self.finished_seen: list[str] = []
        self.protocol_violations: list[str] = []

    # -------- 执行侧的两步协议 --------

    def execute_model(self, scheduler_output):
        self._check_protocol(scheduler_output)
        self._apply_new_requests(scheduler_output.scheduled_new_reqs)
        self._apply_cached_requests(scheduler_output.scheduled_cached_reqs)
        for req_id in scheduler_output.finished_req_ids:
            self.finished_seen.append(req_id)
            self.req_states.pop(req_id, None)          # 结束清理：镜像是执行侧自己的状态

        if scheduler_output.total_num_scheduled_tokens == 0:
            # 空轮：**不碰模型**，直接回空结果（真实 Runner 也是这么处理的）
            self.pending = None
            return ModelRunnerOutput.make_empty()

        self.pending = scheduler_output
        self.num_forward_calls += 1
        self.num_forward_tokens += scheduler_output.total_num_scheduled_tokens
        return None                                    # 状态存下，等 sample_tokens() 消费

    def sample_tokens(self, grammar_output):
        if self.pending is None:
            return ModelRunnerOutput.make_empty()
        pending, self.pending = self.pending, None

        req_ids: list[str] = []
        sampled_token_ids: list[list[int]] = []
        for req_id, num_scheduled in pending.num_scheduled_tokens.items():
            state = self.req_states[req_id]
            state.num_computed_tokens += num_scheduled      # 与 Scheduler 各自推进到同一位置
            if state.num_computed_tokens < len(state.all_token_ids):
                # 还没算到已知历史的末尾（中间 prefill 块）：本轮不产出
                sampled_token_ids.append([])
            else:
                token = self._next_token(req_id, state)
                state.emitted.append(token)
                sampled_token_ids.append([token])
            req_ids.append(req_id)

        if self.reverse_rows:
            # 故意打乱返回顺序：Scheduler 必须靠 req_id_to_index 取值，不能按自己的顺序 zip
            order = list(reversed(range(len(req_ids))))
            req_ids = [req_ids[i] for i in order]
            sampled_token_ids = [sampled_token_ids[i] for i in order]

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
        )

    def take_draft_token_ids(self):
        return None                                    # 57A 不投机

    # -------- 协议数据的应用 --------

    def _apply_new_requests(self, new_reqs) -> None:
        for data in new_reqs:
            self.req_states[data.req_id] = RunnerRequestState(
                prompt_token_ids=list(data.prompt_token_ids),
                num_computed_tokens=data.num_computed_tokens,
                block_ids=tuple(list(group) for group in data.block_ids),
            )

    def _apply_cached_requests(self, cached) -> None:
        for index, req_id in enumerate(cached.req_ids):
            state = self.req_states.get(req_id)
            if state is None:
                # 执行侧没有它的状态（例如刚被重建过）：用整段历史重建。
                # 这正是 `all_token_ids` 存在的理由——但只在"上一轮没被调度"时才带上。
                all_token_ids = cached.all_token_ids.get(req_id)
                if all_token_ids is None:
                    self.protocol_violations.append(
                        f"{req_id}: 执行侧没有镜像，包也没带 all_token_ids，无法重建")
                    continue
                state = RunnerRequestState(prompt_token_ids=list(all_token_ids))
                self.req_states[req_id] = state

            all_token_ids = cached.all_token_ids.get(req_id)
            if all_token_ids is not None:
                # 重新对齐整段历史：prompt 之后的都算"执行侧记住的输出"
                state.emitted = list(all_token_ids[len(state.prompt_token_ids):])

            # 包里的进度是**旧值**（本轮计算前），用它校正镜像
            state.num_computed_tokens = cached.num_computed_tokens[index]
            # 去掉未提交的尾部：执行侧曾经产出、但最终没被提交的那些
            del state.emitted[cached.num_output_tokens[index]:]
            new_blocks = cached.new_block_ids[index]
            if new_blocks is not None:
                if req_id in cached.resumed_req_ids:
                    # 恢复：整张块表被替换（不是在旧表后面接）
                    state.block_ids = tuple(list(group) for group in new_blocks)
                else:
                    # 普通续跑：这是**新增**块，追加到旧表后面
                    state.block_ids = tuple(
                        list(old) + list(new)
                        for old, new in zip(state.block_ids, new_blocks))

    # -------- 协议违规检查 --------

    def _check_protocol(self, scheduler_output) -> None:
        """跨边界的包里不许有活对象、内部类型或闭包。发现就记录（测试会断言它是空的）。"""
        seen = []

        def walk(value, path):
            if isinstance(value, Request):
                seen.append(f"{path} 是内部 Request 活对象")
            elif callable(value):
                seen.append(f"{path} 是可调用对象（闭包回调）")
            elif hasattr(value, "scheduler") or hasattr(value, "kv_cache_pool"):
                seen.append(f"{path} 像是 Scheduler / KV 池这类活对象")
            elif isinstance(value, dict):
                for key, item in value.items():
                    walk(item, f"{path}.{key}")
            elif isinstance(value, (list, tuple, set)):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")

        for name, value in vars(scheduler_output).items():
            walk(value, name)
        self.protocol_violations.extend(seen)

    # -------- token 来源 --------

    def _next_token(self, req_id: str, state: RunnerRequestState) -> int:
        script = self.tokens.get(req_id, [])
        position = len(state.emitted)
        if position >= len(script):
            raise AssertionError(
                f"FakeRunner 的脚本用完了：请求 {req_id!r} 第 {position + 1} 枚 token 没有脚本。"
                f"本关要求测试显式给出输出（不提供看不见的伪随机兜底）；"
                f"当前脚本={script}")
        return script[position]
