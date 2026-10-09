"""调度 → 执行的**数据包**（对应 vLLM `v1/core/sched/output.py`）。

这一层是本关的硬约束所在：**跨执行边界只传状态快照和结果**。包里有：

    SchedulerOutput
      ├─ scheduled_new_reqs   首次被执行的请求（只在第一次发，之后只发增量）
      ├─ scheduled_cached_reqs 续跑/恢复的请求：新增块、旧进度快照、需要时的完整 token 历史
      ├─ num_scheduled_tokens  req_id → 本轮算几个 token
      ├─ scheduled_spec_decode_tokens  57E 才填
      └─ finished_req_ids     要执行侧删掉缓存的请求

**明确不能出现**：`request` / `seq` / `scheduler` / `kv_cache_pool` 这类活对象，也不能塞闭包
回调。三条理由：Worker 将来可能在子进程（活对象不能序列化）、Scheduler 可以被单独测试（不
需要造模型）、以及"谁改了什么"必须一眼可查。

快照还要**复制可变容器**：`all_token_ids` 用 `.copy()`、块表用新 list。执行侧改了包里的列表
不能影响 Scheduler 手里的 Request——这条有专门用例（`check_step57_scheduler_basic.py`）。
"""

from dataclasses import dataclass, field

from ...sampling_params import SamplingParams


@dataclass
class NewRequestData:
    """首次被执行的请求。之后每轮只发增量（`CachedRequestData`），不重复发整段 prompt。

    `block_ids` 的第一维是 **KV group**（本关只有一个 group，所以是 `(group0,)`），
    不是模型层——层是模型侧自己的事。
    """

    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int          # 本轮计算**之前**的快照
    #: 73 关（V2）：要喂进 runner 的**全部** token（= prompt + 抢占恢复时需要重算的已生成 token）。
    #: V1 从"自己镜像里的历史 + prompt"拼出来，所以这条为空；V2 的常驻状态由这条一次性建好
    #: （上游 `NewRequestData.prefill_token_ids`，`scheduler.py:1194-1204`）。
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(cls, request, block_ids: tuple[list[int], ...],
                     prefill_token_ids: list[int] | None = None) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            # 复制：执行侧改这个列表不能影响 Scheduler 手里的 Request
            prompt_token_ids=list(request.prompt_token_ids),
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            prefill_token_ids=(None if prefill_token_ids is None
                               else list(prefill_token_ids)),
        )


@dataclass
class CachedRequestData:
    """续跑（含恢复）的请求。列表按 `req_ids` 对齐。

    - `new_block_ids`：普通续跑是**新增**块；`resumed_req_ids` 里的请求表示**重建后的整张**
      块表（不是接在旧表后面）。
    - `num_computed_tokens`：本轮开始位置，执行侧用它校正自己的镜像。
    - `num_output_tokens`：校正执行侧已保存的输出前缀（去掉未提交的尾部）。
    - `all_token_ids`：只在"上一轮没被调度"时带上，供执行侧重建批状态；**不是每轮复制全部
      历史**（vLLM 的注释同样强调这点）。
    """

    req_ids: list[str]
    resumed_req_ids: set[str]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]
    all_token_ids: dict[str, list[int]]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(req_ids=[], resumed_req_ids=set(), new_block_ids=[],
                   num_computed_tokens=[], num_output_tokens=[], all_token_ids={})


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData

    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]      # 57E 才非空
    finished_req_ids: set[str]
    #: 68 关：这一轮有没有"已经在解码阶段的结构化输出请求"。没有的话引擎连掩码都不必算
    #: （上游同名字段，在 `_update_after_schedule()` 里置位）。
    has_structured_output_requests: bool = False
    #: 71 关（动态投机长度）：本轮调度器选出的 K——执行侧要按它提议**下一轮**的草稿
    #: （上游 `SchedulerOutput.num_spec_tokens_to_schedule`，`output.py:267-269`）。
    #: 默认 0 = "这条路径没有投示意图"（`make_empty()` 与手工构造的非投机包）；引擎路径上
    #: `Scheduler.schedule()` **总会**填（静态 K 时就是配置的 K）。
    #: ⚠️ 它与 `scheduled_spec_decode_tokens` 的长度的含义不同：那个是"本轮要**验证**的旧候选"
    #: （上一轮提的，可能被预算截短），这个是"本轮要**提**的新草稿宽度"。
    num_spec_tokens_to_schedule: int = 0

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(scheduled_new_reqs=[], scheduled_cached_reqs=CachedRequestData.make_empty(),
                   num_scheduled_tokens={}, total_num_scheduled_tokens=0,
                   scheduled_spec_decode_tokens={}, finished_req_ids=set())


@dataclass
class GrammarOutput:
    """Scheduler → 执行侧：这一轮的语法掩码（对应 vLLM `v1/core/sched/output.py::GrammarOutput`）。

    **只有两个字段**（上游同款）：
        structured_output_request_ids  掩码按这个顺序排列（不是批行序！）
        grammar_bitmask                `[num_masks, ceil(vocab/32)]` 的 numpy int32

    顺序为什么重要：掩码是"每个待定位置一行"，而"位置 → logits 行号"的映射还要叠上每请求
    的草稿数；执行侧必须按同一份 id 列表重建映射，不能假设它等于批行序。
    """

    structured_output_request_ids: list[str]
    grammar_bitmask: object          # np.ndarray[np.int32]
