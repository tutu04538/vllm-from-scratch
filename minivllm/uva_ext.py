"""UVA 设备视图的唯一入口（对应上游 C++ 算子 `_C.get_cuda_view_from_cpu_tensor`）。

上游的算子就三步（从已安装的 `.so` 符号读出来，见 `docs/step73_alignment.md` §5 D2）：
`aten::is_pinned` 检查 → `cudaHostGetDevicePointer` → `torch::stable::from_blob(..., kCUDA)`。
本仓库没有预编译扩展，于是给出**两条实现**，用环境变量 `MINIVLLM_UVA_BACKEND` **显式**选择：

    cpp（默认）  用 `torch.utils.cpp_extension.load_inline` 现编一个同构算子（约 35 行 C++）。
                 机制与上游同级：复用 torch 的 pinned 分配器、张量原生存周期、没有 DLPack 手工包装。
    python       ctypes 调同两个 CUDA runtime 函数 + DLPack 包装（无需 C++ 工具链的兜底）。

**为什么不自动回退**：静默换实现会让"与上游对齐"变成口号（用户看到的是同一条路径、
跑的却是另一套机制）。所以默认走 cpp，构建失败就**带着原因报错**，并在错误里直接给出
`MINIVLLM_UVA_BACKEND=python` 这个显式开关。

首次导入会冷编译（实测 35.1 s，之后走 torch 的扩展缓存目录，毫秒级）；这与本仓库
Triton 内核的运行时编译是同一类依赖（都要 CUDA toolkit），不是新引入的构建体系。
"""

import os

import torch

_BACKENDS = ("cpp", "python")
_DEFAULT_BACKEND = "cpp"

# 编译产物缓存在这里（属于 torch 的扩展缓存，不进仓库、不当交付物）
_EXT_NAME = "minivllm_uva_view"

_ext = None  # 懒加载：只有真的要用 cpp 后端时才编译


def resolve_backend() -> str:
    """读 `MINIVLLM_UVA_BACKEND`（每次调用都读：测试要在同一进程里切换后端）。"""
    value = os.environ.get("MINIVLLM_UVA_BACKEND", _DEFAULT_BACKEND).strip().lower()
    if value not in _BACKENDS:
        raise ValueError(
            f"MINIVLLM_UVA_BACKEND 只能是 {_BACKENDS} 之一，收到 {value!r}"
            "（cpp = 现编上游同构算子；python = ctypes + DLPack 兜底）"
        )
    return value


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// 与上游 `_C.get_cuda_view_from_cpu_tensor` 三步同构：is_pinned 检查 → 取设备指针 → from_blob。
// 不做 cudaHostRegister：torch 的 pinned 分配器分配的内存本身就是 UVA 映射的（本机实测
// cudaHostGetDevicePointer 直接成功且返回同一个虚拟地址），上游的算子也依赖这一点。
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor cpu_tensor) {
  TORCH_CHECK(cpu_tensor.is_pinned(), "CPU tensor must be pinned");
  void* host_ptr = cpu_tensor.data_ptr();
  void* dev_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
  TORCH_CHECK(err == cudaSuccess,
              "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  auto options = cpu_tensor.options().device(torch::kCUDA);
  // 借用内存（不接管生命周期）：宿主张量由调用方持有，与上游同款约定。
  return torch::from_blob(dev_ptr, cpu_tensor.sizes(), cpu_tensor.strides(), options);
}
"""


def _load_ext():
    """现编并缓存那个算子。构建失败 → 明确报错（附兜底开关）。"""
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load_inline

    try:
        _ext = load_inline(
            name=_EXT_NAME,
            cpp_sources=("torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor cpu_tensor);"),
            cuda_sources=_CUDA_SOURCE,
            functions=["get_cuda_view_from_cpu_tensor"],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 —— 任何构建失败都要说清"怎么办"
        raise RuntimeError(
            "UVA 设备视图的 C++ 扩展构建失败（需要 nvcc + CUDA toolkit + torch 头文件）："
            f"{type(exc).__name__}: {exc}。"
            "想在没有工具链的机器上跑，请显式设 MINIVLLM_UVA_BACKEND=python 走 ctypes/DLPack 兜底"
            "（本仓库不会自动回退：静默换实现等于两条路径都没对齐）"
        ) from exc
    return _ext


def get_cuda_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """pinned CPU 张量 → **别名同一块内存** 的 CUDA 张量（共享内存，不是拷贝）。"""
    if resolve_backend() == "cpp":
        return _load_ext().get_cuda_view_from_cpu_tensor(cpu_tensor)
    from .utils.torch_utils import get_accelerator_view_from_cpu_tensor

    return get_accelerator_view_from_cpu_tensor(cpu_tensor)
