"""82 关纯统计工具：累计输出去重、接受率、TTFT/TPOT、吞吐。

这个文件不 import minivllm 引擎，只接收 81 关的 SpecDecodingStats 与逐请求时间戳，
方便 `tests/step82/test_benchmark_metrics.py` 用小整数验证公式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass
class RequestTiming:
    request_id: str
    submitted_at: float
    first_output_at: float | None = None
    last_output_at: float | None = None
    committed_tokens: int = 0
    output_events: list[tuple[float, int]] = field(default_factory=list)

    def observe(self, now: float, total_tokens: int) -> int:
        """吃累计 token 数，返回**本步新增**数；重复累计值不会重复加。"""
        increment = max(total_tokens - self.committed_tokens, 0)
        if increment:
            self.committed_tokens = total_tokens
            if self.first_output_at is None:
                self.first_output_at = now
            self.last_output_at = now
            self.output_events.append((now, increment))
        return increment


@dataclass
class AcceptanceAggregate:
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    accepted_per_pos: list[int] = field(default_factory=list)
    draft_per_pos: list[int] = field(default_factory=list)

    def observe(self, stats) -> None:
        """吃 81 的 `SpecDecodingStats`（不要求 import 其类）。"""
        self.num_drafts += int(stats.num_drafts)
        self.num_draft_tokens += int(stats.num_draft_tokens)
        self.num_accepted_tokens += int(stats.num_accepted_tokens)
        while len(self.accepted_per_pos) < len(stats.num_accepted_tokens_per_pos):
            self.accepted_per_pos.append(0)
            self.draft_per_pos.append(0)
        for i, value in enumerate(stats.num_accepted_tokens_per_pos):
            self.accepted_per_pos[i] += int(value)
        for i, value in enumerate(stats.num_draft_tokens_per_pos):
            self.draft_per_pos[i] += int(value)

    @property
    def acceptance_rate(self) -> float | None:
        if self.num_draft_tokens == 0:
            return None
        return self.num_accepted_tokens / self.num_draft_tokens

    @property
    def mean_acceptance_length(self) -> float | None:
        if self.num_drafts == 0:
            return None
        return 1.0 + self.num_accepted_tokens / self.num_drafts

    def position_acceptance(self) -> list[float | None]:
        """上游日志口径：accepted[j] / num_drafts（不是 / draft_per_pos[j]）。"""
        if self.num_drafts == 0:
            return []
        return [value / self.num_drafts for value in self.accepted_per_pos]

    def position_acceptance_actual(self) -> list[float | None]:
        """实际出现位置口径：accepted[j] / draft_per_pos[j]；零分母给 None。"""
        out: list[float | None] = []
        for accepted, drafted in zip(self.accepted_per_pos, self.draft_per_pos):
            out.append(accepted / drafted if drafted else None)
        return out


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    weight = position - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summarize_run(*, timings: dict[str, RequestTiming], acceptance: AcceptanceAggregate,
                  wall_time_s: float, engine_steps: int,
                  preemptions: int = 0) -> dict:
    """把一个 workload 的原始计时/接受统计变成一个 JSON 友好的结果。"""
    committed = sum(t.committed_tokens for t in timings.values())
    ttfts = [t.first_output_at - t.submitted_at
             for t in timings.values() if t.first_output_at is not None]
    tpots = []
    bursts = []
    for timing in timings.values():
        if len(timing.output_events) >= 2:
            tpots.append((timing.last_output_at - timing.first_output_at)
                         / (len(timing.output_events) - 1))
        bursts.extend(count for _, count in timing.output_events)
    return {
        "requests": len(timings),
        "committed_output_tokens": committed,
        "wall_time_s": wall_time_s,
        "output_tokens_per_s": (committed / wall_time_s if wall_time_s > 0 else None),
        "ttft_s": {
            "mean": mean(ttfts) if ttfts else None,
            "p50": _percentile(ttfts, 0.5),
            "min": min(ttfts) if ttfts else None,
            "max": max(ttfts) if ttfts else None,
        },
        "tpot_s": {
            "mean": mean(tpots) if tpots else None,
            "p50": _percentile(tpots, 0.5),
        },
        "burst_tokens": {
            "mean": mean(bursts) if bursts else None,
            "max": max(bursts) if bursts else None,
        },
        "engine_steps": engine_steps,
        "preemptions": preemptions,
        "acceptance": {
            "num_drafts": acceptance.num_drafts,
            "num_draft_tokens": acceptance.num_draft_tokens,
            "num_accepted_tokens": acceptance.num_accepted_tokens,
            "acceptance_rate": acceptance.acceptance_rate,
            "mean_acceptance_length": acceptance.mean_acceptance_length,
            "accepted_per_pos": acceptance.accepted_per_pos,
            "draft_per_pos": acceptance.draft_per_pos,
            "position_acceptance": acceptance.position_acceptance(),
            "position_acceptance_actual": acceptance.position_acceptance_actual(),
        },
    }


def median_range(values: list[float]) -> dict:
    if not values:
        return {"median": None, "min": None, "max": None, "values": []}
    return {"median": mean(values) if len(values) == 1 else _percentile(values, 0.5),
            "min": min(values), "max": max(values), "values": list(values)}
