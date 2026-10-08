"""CUDA Graph 的命中统计与汇总表（对应 vLLM `compilation/cuda_graph.py::CUDAGraphStat/Logging`）。

**它解决什么**：开了图之后，"这一步到底走了图没有、走的是哪一档、补了多少行"是看不见的。
没有这份账，验收只能看跑得快不快——而"没走图"和"走了图但更慢"在时间上分不出来。
上游把它做成一份按 (未补齐行数, 补齐行数, 模式) 聚合的计数表，随日志周期性打出；
本仓库同样实现这张表，另外把原始记录挂在 Runner 上，测试与 profiler 直接读（不靠解析日志）。

字段与上游同名同义（`CUDAGraphStat`）：

    num_unpadded_tokens   这一轮真实的行数（调度排了多少行）
    num_padded_tokens     补齐到档位之后的行数（图实际跑的形状）
    num_paddings          补齐多出来的行数
    runtime_mode          这一轮真实选中的运行模式名（NONE / PIECEWISE / FULL）

**与上游的差异**：本仓库不做"周期性自动打日志"（没有 metrics 前端与日志节流配置），
只提供 `generate_metric_table()` 与 `log()`，由调用方（探针/测试/demo）决定何时打。
"""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class CUDAGraphStat:
    """一轮的图命中记录（上游同名 dataclass，字段逐字对齐）。"""

    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str


class CUDAGraphLogging:
    """把每轮的记录聚合成一张可读的表（上游同名类，`generate_metric_table()` 逐段对齐）。"""

    COLUMN_HEADERS = ["Unpadded Tokens", "Padded Tokens", "Num Paddings",
                      "Runtime Mode", "Count"]

    def __init__(self, cg_mode, cg_capture_sizes) -> None:
        self.reset()
        self.cg_mode = str(cg_mode)
        self.cg_capture_sizes = str(list(cg_capture_sizes or []))
        self.settings_header = (
            "**CUDAGraph 配置：**\n\n"
            f"- 模式: {self.cg_mode}\n"
            f"- 档位表: {self.cg_capture_sizes}\n\n"
            "**CUDAGraph 命中统计：**\n\n"
        )

    def reset(self) -> None:
        self.stats: list[CUDAGraphStat] = []

    def observe(self, cudagraph_stat: CUDAGraphStat) -> None:
        self.stats.append(cudagraph_stat)

    def generate_metric_table(self) -> str:
        stats_counts = Counter(self.stats)
        rows = []
        for stat, count in sorted(stats_counts.items(), key=lambda item: item[1],
                                  reverse=True):
            rows.append([str(stat.num_unpadded_tokens), str(stat.num_padded_tokens),
                         str(stat.num_paddings), stat.runtime_mode, str(count)])

        col_widths = []
        for i, header_text in enumerate(self.COLUMN_HEADERS):
            max_width = len(header_text)
            for row in rows:
                max_width = max(max_width, len(row[i]))
            col_widths.append(max_width)

        table_header = "| " + " | ".join(
            h.ljust(w) for h, w in zip(self.COLUMN_HEADERS, col_widths)) + " |\n"
        table_separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n"
        data_rows = ["| " + " | ".join(
            val.ljust(width) for val, width in zip(row, col_widths)) + " |"
            for row in rows]
        return (self.settings_header + table_header + table_separator
                + "\n".join(data_rows) + "\n")

    def log(self, log_fn=None) -> None:
        """打表并清空（上游同款）；调用方给 log 函数，默认用模块 logger。"""
        if not self.stats:
            return
        if log_fn is None:
            import logging

            log_fn = logging.getLogger(__name__).info
        log_fn(self.generate_metric_table())
        self.reset()
