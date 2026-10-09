"""torch 小工具（对应 vLLM `vllm/utils/torch_utils.py` 的子集，只搬 V2 通路要用的三个）。

- `async_tensor_h2d`：list/ndarray/tensor → device，**非阻塞**；pinned 与否由
  `is_pin_memory_available()` 决定（上游用模块级常量 `PIN_MEMORY`，语义相同）。
- `np_to_pinned_tensor`：numpy → pinned tensor。
- `get_accelerator_view_from_cpu_tensor`：pinned CPU 张量的**设备视图**（UVA 的核心 API：
  同一块物理内存既能被 CPU 写、又能被 GPU 读，所以"CPU 侧改一份、GPU 侧立刻能看见"）。
"""

import numpy as np
import torch

from .platform_utils import is_pin_memory_available


def async_tensor_h2d(
    data: list | np.ndarray | torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """把 list/numpy/tensor 异步拷到 device（上游 `utils/torch_utils.py:690` 同款）。"""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        t = data.pin_memory() if is_pin_memory_available() else data
    else:
        t = torch.tensor(
            data, dtype=dtype, pin_memory=is_pin_memory_available(), device="cpu"
        )
    assert t.is_cpu
    return t.to(device=device, dtype=dtype, non_blocking=True)


def np_to_pinned_tensor(array: np.ndarray) -> torch.Tensor:
    """numpy → pinned CPU 张量（pinned 不可用时就是普通张量，与上游同）。"""
    t = torch.from_numpy(array)
    return t.pin_memory() if is_pin_memory_available() else t


def get_accelerator_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """pinned CPU 张量的设备视图（UVA）。上游只对 XPU 做特判，本仓库只有 CUDA/CPU。"""
    assert cpu_tensor.is_pinned(), "UVA 视图要求 CPU 张量是 pinned 的"
    return cpu_tensor.to("cuda", non_blocking=True)


#: 配置里的 dtype 字符串 → torch dtype（对应上游 `vllm/utils/torch_utils.py:33` 的同名表）。
#: 为什么要有它：本仓库的 `ModelConfig.dtype` 是**字符串**（配置里写 `"bfloat16"` 这种），
#: 而 V2 的提议者要在 `load_model()` **之前**用 dtype 开常驻缓冲（上游那里已经是 torch.dtype）。
STR_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "float64": torch.float64,
}
