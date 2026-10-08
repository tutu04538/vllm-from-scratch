"""编译与 CUDA Graph（对应 vLLM `vllm/compilation/` 的子集）。

本仓库**没有编译路径**（`CompilationMode` 只支持 NONE，配置期明确拒绝其余取值），
这里只有 CUDA Graph 那一半：

    monitor.py      "现在允许捕获吗"的全局开关（捕获窗口）
    cuda_graph.py   图包装器：读 forward 上下文 → 直通 / 捕获 / 重放

为什么要分成"图"与"编译"两件事：编译（torch.compile）改的是**算子怎么算**（融合、少 launch
几个小 kernel），图改的是**kernel 怎么发**（一次提交、不再逐 kernel 过 Python/驱动的启动
开销）。二者独立，可以只做图——本仓库就是这条（`mode=none` + `cudagraph_mode=full`）。
"""

from .cuda_graph import (CUDAGraphEntry, CUDAGraphOptions, CUDAGraphWrapper, graph_capture)
from .monitor import set_cudagraph_capturing_enabled, validate_cudagraph_capturing_enabled
from .stats import CUDAGraphLogging, CUDAGraphStat

__all__ = ["CUDAGraphEntry", "CUDAGraphLogging", "CUDAGraphOptions", "CUDAGraphStat",
           "CUDAGraphWrapper", "graph_capture",
           "set_cudagraph_capturing_enabled", "validate_cudagraph_capturing_enabled"]
