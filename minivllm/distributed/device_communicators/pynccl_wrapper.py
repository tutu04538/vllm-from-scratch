"""NCCL C ABI used by the local PyNccl communicator.

The function signatures and enum values follow frozen vLLM's pynccl_wrapper.
"""
from __future__ import annotations

import ctypes
import os

import torch


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


_DTYPES = {
    torch.int8: 0, torch.uint8: 1, torch.int32: 2, torch.int64: 4,
    torch.float16: 6, torch.float32: 7, torch.float64: 8, torch.bfloat16: 9,
}


def nccl_dtype(dtype: torch.dtype) -> int:
    try:
        return _DTYPES[dtype]
    except KeyError as exc:
        raise ValueError(f"NCCL 不支持的数据类型 {dtype}") from exc


class NCCLLibrary:
    def __init__(self):
        self.path = os.environ.get("VLLM_NCCL_SO_PATH", "libnccl.so.2")
        self.lib = ctypes.CDLL(self.path)
        ptr = ctypes.c_void_p
        signatures = {
            "ncclGetUniqueId": [ctypes.POINTER(NcclUniqueId)],
            "ncclCommInitRank": [ctypes.POINTER(ptr), ctypes.c_int, NcclUniqueId, ctypes.c_int],
            "ncclAllReduce": [ptr, ptr, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ptr, ptr],
            "ncclAllGather": [ptr, ptr, ctypes.c_size_t, ctypes.c_int, ptr, ptr],
            "ncclCommAbort": [ptr],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = ctypes.c_int, args
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p
        self.lib.ncclGetErrorString.argtypes = [ctypes.c_int]

    def call(self, name, *args):
        error = getattr(self.lib, name)(*args)
        if error:
            message = self.lib.ncclGetErrorString(error).decode("utf-8", "replace")
            raise RuntimeError(f"{name} NCCL error: {message}")
