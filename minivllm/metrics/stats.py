"""81 关最小 `SchedulerStats`（对应 vLLM `v1/metrics/stats.py` 的子集）。

只保留投机主线实际消费的字段：
  - 运行/等待请求数（便于判断清理轮/空轮）；
  - 本步 `SpecDecodingStats`（没有投机或本轮没有验证时为 None）。

不移植 DP、LoRA、connector、prefix-cache 统计等未使用字段。
"""

from dataclasses import dataclass

from ..spec_decode.metrics import SpecDecodingStats


@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    spec_decoding_stats: SpecDecodingStats | None = None
