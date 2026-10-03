"""`SpecDecodingStats`：投机一轮的接受情况统计（对应 vLLM `v1/spec_decode/metrics.py`）。

**只统计"已经验证过"的候选**（059 §2）：一条请求本轮排了 K 枚草稿、target 逐位置验完，
才把 K 与"接受的枚数"记进来。**提议数不能当接受数**——提议是在上一轮做的，这一轮可能只采用
了它的前缀（预算截断），甚至一枚都没采用；拿提议数记进来会把接受率算高，进而把"第 j 枚草稿
还值不值得提"判断错。

上游把它按 `SchedulerStats` 汇总后送到前端（`SpecDecodingLogging` 打印日志 + prometheus 指标）；
本项目没有指标前端，所以 Scheduler 只保留**每一步的这一份**（`Scheduler.spec_decoding_stats`），
测试与 demo 直接读它。

各字段的含义（与上游逐字一致）：

    num_spec_tokens             配置的 K（每请求最多提几枚）
    num_drafts                  本步被统计的请求数（"轮次"数）
    num_draft_tokens            被验证的候选总数（ΣK_i）
    num_accepted_tokens         被接受的候选总数（Σa_i）
    num_accepted_tokens_per_pos 第 j 个位置上被接受的次数（j < K）
    num_draft_tokens_per_pos    第 j 个位置上被验证的次数（j < K）

两个 per_pos 数组是做"位置级接受率"用的：`accepted[j] / drafts[j]` 就是第 j 枚草稿的接受率，
它随 j 快速衰减——这个数决定 K 该取多大（每关只做小规模测量，性能矩阵在 84 关）。
"""

from dataclasses import dataclass, field


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_draft_tokens_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_draft_tokens_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int) -> None:
        """记下一条请求这一轮的 (验证了几枚, 接受了几枚)。"""
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        # 接受的枚数不可能超过"配置最多提几枚"：超了说明上游把未验证的也算进来了
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
        for i in range(num_draft_tokens):
            self.num_draft_tokens_per_pos[i] += 1
