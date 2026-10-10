"""torch 小工具（对应 vLLM `vllm/utils/torch_utils.py` 的子集，只搬 V2 通路要用的三个）。

- `async_tensor_h2d`：list/ndarray/tensor → device，**非阻塞**；pinned 与否由
  `is_pin_memory_available()` 决定（上游用模块级常量 `PIN_MEMORY`，语义相同）。
- `np_to_pinned_tensor`：numpy → pinned tensor。
- `get_accelerator_view_from_cpu_tensor`：pinned CPU 张量的**设备视图**（UVA 的核心 API：
  同一块物理内存既能被 CPU 写、又能被 GPU 读，所以"CPU 侧改一份、GPU 侧立刻能看见"）。
"""

import ctypes
import ctypes.util

import numpy as np
import torch

from .platform_utils import is_pin_memory_available


# ---------------------------------------------------------------------------
# DLPack 结构体（python 兜底实现包设备指针用；对应 `dlpack.h` 的那几个 struct）
# ---------------------------------------------------------------------------


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DELETER_T = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER_T),
]

#: DLPack 的 dtype code：kDLInt=0、kDLFloat=2
_DL_CODES = {torch.int32: (0, 32), torch.int64: (0, 64), torch.float32: (2, 32)}


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


# ---------------------------------------------------------------------------
# UVA 设备视图的 **python 兜底实现**（默认走 `minivllm/uva_ext.py` 的 C++ 算子）
# ---------------------------------------------------------------------------

#: DLPack 的 deleter 必须是 **NULL**（不是 Python 回调）。实测：张量在解释器退出阶段析构时
#: torch 会回调 deleter，而那时 ctypes 的 CFUNCTYPE 弹簧床已被拆掉 → 段错误（exit=139）。
#: DLPack 允许 `deleter == NULL`（表示内存由生产者持有），映射内存本来就是进程级不释放的。
_DELETER_T = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DL_DELETER = ctypes.cast(None, _DELETER_T)

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

#: 够住 DLPack 结构体与形状数组：它们必须活到张量析构（进程结束前不释放）。
_DLPACK_KEEPALIVE: list = []


def _cudart():
    lib = ctypes.util.find_library("cudart")
    if lib is None:
        raise RuntimeError(
            "找不到 libcudart：UVA 的 python 兜底实现需要它（或改用 MINIVLLM_UVA_BACKEND=cpp）"
        )
    return ctypes.CDLL(lib)


def get_accelerator_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """pinned CPU 张量的设备视图（UVA）——**python 兜底实现**。

    与上游 C++ 算子做同样的事：`is_pinned` 检查 → `cudaHostGetDevicePointer` → 包成 CUDA 张量
    （这里用 DLPack，因为本项目没有 `torch::stable::from_blob` 那层 C++）。返回的张量**别名**
    同一块主机内存：GPU 内核写进去、CPU 立刻能读到；CPU 改一行、内核也立刻看到。
    """
    assert cpu_tensor.is_pinned(), "UVA 视图要求 CPU 张量是 pinned 的"
    assert cpu_tensor.is_contiguous(), "UVA 视图要求连续张量（DLPack 走 strides=None 的紧凑布局）"
    lib = _cudart()
    lib.cudaHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    lib.cudaHostGetDevicePointer.restype = ctypes.c_int
    dev = ctypes.c_void_p()
    err = lib.cudaHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(cpu_tensor.data_ptr()), 0)
    if err != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer 失败：cudaError={err}")

    code, bits = _DL_CODES[cpu_tensor.dtype]
    shape = (ctypes.c_int64 * cpu_tensor.dim())(*cpu_tensor.shape)
    mt = _DLManagedTensor()
    mt.dl_tensor = _DLTensor(
        ctypes.c_void_p(int(dev.value)),
        _DLDevice(2, cpu_tensor.device.index or 0),   # kDLCUDA
        cpu_tensor.dim(),
        _DLDataType(code, bits, 1),
        shape,
        None,
        0,
    )
    mt.manager_ctx = None
    mt.deleter = _DL_DELETER
    capsule = _PyCapsule_New(ctypes.cast(ctypes.pointer(mt), ctypes.c_void_p), b"dltensor", None)

    class _UvaView:
        def __dlpack__(self, stream=None):
            return capsule

        def __dlpack_device__(self):
            return (2, cpu_tensor.device.index or 0)

    holder = _UvaView()
    _DLPACK_KEEPALIVE.append((shape, mt, holder, capsule, cpu_tensor))
    return torch.from_dlpack(holder)


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
