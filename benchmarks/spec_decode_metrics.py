"""82/83 关纯统计工具：累计输出去重、接受率、TTFT、burst 间隔、TPOT、吞吐。

这个文件不 import minivllm 引擎，只接收 81 关的 SpecDecodingStats 与逐请求时间戳，
方便 `tests/step82`、`tests/step83` 用小整数验证公式。

**83 关指标口径**（历史结果文件不重写）：

- `output_tokens_per_s = committed_output_tokens / wall_time_s`，只数最终提交的输出 token，
  不把草稿/验证候选算进去；
- `inter_burst_latency_s` 是旧 `tpot_s` 的准确名字：按**输出事件个数**做分母，只在有至少
  两个 burst 时有定义；
- `tpot_s` 从 83 起按**首个 burst 之后的实际输出 token 数**做分母。单 burst 时为
  `None`（不可定义），不能填 0。例如 t=1s 返回 3 token、t=2s 返回 2 token：
  burst 间隔 = 1s，TPOT = 1s / 2 token = 0.5 s/token；
- TTFT、接受率、每位置接受数各自独立；验证阶段统计与最终截断后的输出 token 不混用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

METRIC_SCHEMA = "step83"


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


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _p50_or_none(values: list[float]) -> float | None:
    return _percentile(values, 0.5)


def _inter_burst_latency_s(timing: RequestTiming) -> float | None:
    """旧 `tpot_s` 的定义，保留为历史可比字段。

    分母是**输出事件数 − 1**，即"相邻两次交付之间的平均间隔"。它既不是每个 token 的
    延迟，也不能推断同一个 burst 内 token 的到达时间。
    """
    if len(timing.output_events) < 2:
        return None
    return ((timing.last_output_at - timing.first_output_at)
            / (len(timing.output_events) - 1))


def _tpot_s(timing: RequestTiming) -> float | None:
    """83 口径：首个 burst 之后每个**实际提交**输出 token 的平均时间。

    首 burst 的 token 已经一次性交付，无法从外部时间戳拆出单个 token 的真实到达时间，
    所以只对"首 burst 之后"的 token 做平均；单 burst -> 不可定义（None），不是 0。
    """
    if len(timing.output_events) < 2:
        return None
    tokens_after_first = timing.committed_tokens - timing.output_events[0][1]
    if tokens_after_first <= 0:
        return None
    return ((timing.last_output_at - timing.first_output_at)
            / tokens_after_first)


def summarize_run(*, timings: dict[str, RequestTiming], acceptance: AcceptanceAggregate,
                  wall_time_s: float, engine_steps: int,
                  preemptions: int = 0) -> dict:
    """把一个 workload 的原始计时/接受统计变成一个 JSON 友好的结果。"""
    committed = sum(t.committed_tokens for t in timings.values())
    ttfts = [t.first_output_at - t.submitted_at
             for t in timings.values() if t.first_output_at is not None]
    inter_burst = []
    tpots = []
    bursts = []
    for timing in timings.values():
        value = _inter_burst_latency_s(timing)
        if value is not None:
            inter_burst.append(value)
        value = _tpot_s(timing)
        if value is not None:
            tpots.append(value)
        bursts.extend(count for _, count in timing.output_events)
    return {
        "metric_schema": METRIC_SCHEMA,
        "requests": len(timings),
        "committed_output_tokens": committed,
        "wall_time_s": wall_time_s,
        "output_tokens_per_s": (committed / wall_time_s if wall_time_s > 0 else None),
        "ttft_s": {
            "mean": _mean_or_none(ttfts),
            "p50": _p50_or_none(ttfts),
            "min": min(ttfts) if ttfts else None,
            "max": max(ttfts) if ttfts else None,
        },
        # 旧定义改名，历史 step82 文件里的 `tpot_s` 不要按这个字段重算。
        "inter_burst_latency_s": {
            "mean": _mean_or_none(inter_burst),
            "p50": _p50_or_none(inter_burst),
            "num_requests": len(inter_burst),
        },
        # 83 起的新 TPOT；单 burst 的请求不参与，字段为 None。
        "tpot_s": {
            "mean": _mean_or_none(tpots),
            "p50": _p50_or_none(tpots),
            "num_requests": len(tpots),
            "undefined_requests": len(timings) - len(tpots),
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


def memory_snapshot() -> dict:
    """当前 CUDA 显存快照；CPU 上返回 `available=False`。

    只把 allocated/reserved/device_total 放在一起，避免把 reserved（缓存池）说成物理驻留。
    调用方负责分阶段：加载/捕获 vs 预热后的正式测量。
    """
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "device_total_bytes": total,
        "device_free_bytes": free,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def reset_peak_memory() -> None:
    """把峰值统计重置到当前状态；只在阶段边界调用，不进入每 step 热路径。"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def median_range(values: list[float]) -> dict:
    if not values:
        return {"median": None, "min": None, "max": None, "values": []}
    return {"median": _percentile(values, 0.5),
            "min": min(values), "max": max(values), "values": list(values)}
