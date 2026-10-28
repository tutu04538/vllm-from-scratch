"""`SpecDecodingStats`：投机一轮的接受情况统计（对应 vLLM `v1/spec_decode/metrics.py`）。

**只统计"已经验证过"的候选**（059 §2）：一条请求本轮排了 K 枚草稿、target 逐位置验完，
才把 K 与"接受的枚数"记进来。**提议数不能当接受数**——提议是在上一轮做的，这一轮可能只采用
了它的前缀（预算截断），甚至一枚都没采用；拿提议数记进来会把接受率算高，进而把"第 j 枚草稿
还值不值得提"判断错。

81 关起按上游方向接到 `SchedulerStats → EngineCoreOutputs → LLMEngine`，
由 `SpecDecodingLogging.observe` 聚合并可 `log/reset`（Prometheus 服务化不在本关范围）。
Scheduler 仍保留**每一步的这一份**（`Scheduler.spec_decoding_stats`）方便旧测试读取。

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

import logging
import time
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


class SpecDecodingLogging:
    """81：把每步 `SpecDecodingStats` 聚合成可日志/可检查的快照（AR 子集）。

    上游还接 Prometheus；本关只保留 `observe/log/reset`。
    `observe` 必须复制 per-position 列表：Scheduler 下一步会新建 stats，但调用方可能
    保留旧 dict 做快照，不能让它被原地改写。
    """

    def __init__(self, log_fn=logging.getLogger(__name__).info) -> None:
        self._log_fn = log_fn
        self.reset()

    def reset(self) -> None:
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.draft_tokens_per_pos_lists: list[list[int]] = []
        self.last_log_time = time.monotonic()

    def observe(self, stats: SpecDecodingStats) -> None:
        self.num_drafts.append(stats.num_drafts)
        self.num_draft_tokens.append(stats.num_draft_tokens)
        self.num_accepted_tokens.append(stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            list(stats.num_accepted_tokens_per_pos))
        # 记住位置分母快照，便于实验统计 accepted[j]/draft_per_pos[j]；
        # 不改变上游 accepted[j]/num_drafts 的日志口径。
        self.draft_tokens_per_pos_lists.append(
            list(stats.num_draft_tokens_per_pos))

    def __len__(self) -> int:
        return len(self.num_drafts)

    def log(self) -> dict:
        """记录聚合值并 reset；返回同一份值，测试/benchmark 不需要解析日志文本。"""
        if not self.num_drafts:
            return {
                "num_drafts": 0,
                "num_draft_tokens": 0,
                "num_accepted_tokens": 0,
                "accepted_tokens_per_pos": [],
                "draft_tokens_per_pos": [],
                "acceptance_rate": None,
                "mean_acceptance_length": None,
            }
        num_drafts = sum(self.num_drafts)
        num_draft_tokens = sum(self.num_draft_tokens)
        num_accepted = sum(self.num_accepted_tokens)
        max_pos = max((len(row) for row in self.accepted_tokens_per_pos_lists),
                      default=0)
        accepted_per_pos = [
            sum(row[j] for row in self.accepted_tokens_per_pos_lists if j < len(row))
            for j in range(max_pos)]
        draft_per_pos = [
            sum(row[j] for row in self.draft_tokens_per_pos_lists if j < len(row))
            for j in range(max_pos)]
        elapsed = max(time.monotonic() - self.last_log_time, 1e-12)
        rate = (num_accepted / num_draft_tokens) if num_draft_tokens else None
        mean_len = (1 + num_accepted / num_drafts) if num_drafts else None
        result = {
            "num_drafts": num_drafts,
            "num_draft_tokens": num_draft_tokens,
            "num_accepted_tokens": num_accepted,
            "accepted_tokens_per_pos": accepted_per_pos,
            "draft_tokens_per_pos": draft_per_pos,
            "acceptance_rate": rate,
            "mean_acceptance_length": mean_len,
            "draft_throughput": num_draft_tokens / elapsed,
            "accepted_throughput": num_accepted / elapsed,
        }
        self._log_fn(
            "SpecDecoding metrics: Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, Drafted: %d tokens, Avg Draft acceptance rate: %s",
            mean_len if mean_len is not None else float("nan"),
            result["accepted_throughput"], result["draft_throughput"],
            num_accepted, num_draft_tokens,
            f"{rate * 100:.1f}%" if rate is not None else "N/A")
        self.reset()
        return result
