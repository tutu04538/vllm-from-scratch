"""triton 导入的唯一入口（对应 vLLM `vllm/triton_utils/__init__.py` 的最小子集）。

为什么要有这个模块（而不是各处直接 `import triton`）：73 关的 V2 通路里有一批**逐行照抄上游**
的内核文件（`worker/gpu/sample/gumbel.py`、`worker/gpu/spec_decode/rejection_sampler_utils.py`），
它们按上游写法从 `vllm.triton_utils` 取 `HAS_TRITON / triton / tl / tldevice`。
本模块就是那组名字（路径映射：`minivllm/` ↔ `vllm/`），这样拷贝过来的内核只需要改 import 前缀，
**不改内核体**——"严格对齐"要求我们连 `tldevice.log1p` 这种写法都不换。

与上游的差异（记在 `docs/step73_alignment.md` 差异账本）：上游在没有 triton 时用
`TritonPlaceholder` 占位（为了让 CPU worker 也能 import）；本仓库的依赖里 triton 是必需项
（59 关起拒绝采样就是 Triton 内核），所以**没有**占位类，triton 缺失时直接 import 失败。
"""

try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover - 本机 triton 3.7.1 常驻
    HAS_TRITON = False
    raise

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453

__all__ = ["HAS_TRITON", "triton", "tl", "tldevice", "LOG2E", "LOGE2"]
