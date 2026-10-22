"""KV cache 量化模式（上游 `vllm/v1/kv_cache_interface.py::KVQuantMode` 的子集）。

本仓库的 TRITON_ATTN 子集只支持 NONE；保留完整 IntEnum 值是为了 vendored kernel
的 `KV_QUANT_MODE` constexpr 与上游同形（未支持的模式由后端显式拒绝）。
"""

from enum import IntEnum


class KVQuantMode(IntEnum):
    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4
    NVFP4 = 5
    TURBOQUANT_K8V4 = 6
    TURBOQUANT_4BIT_NC = 7
    TURBOQUANT_K3V4_NC = 8
    TURBOQUANT_3BIT_NC = 9
