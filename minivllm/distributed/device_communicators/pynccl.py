"""Process-group-owned NCCL communicator, callable inside CUDA Graph capture."""
from __future__ import annotations

import ctypes
import threading

import torch
import torch.distributed as dist

from .pynccl_wrapper import NCCLLibrary, NcclUniqueId, nccl_dtype


class PyNcclCommunicator:
    def __init__(self, group, device: torch.device):
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.world_size <= 1 or dist.get_backend(group) == "nccl":
            raise ValueError("PyNccl 需要多 rank 的 CPU 通信组")
        self.nccl = NCCLLibrary()
        unique_id = NcclUniqueId()
        if self.rank == 0:
            self.nccl.call("ncclGetUniqueId", ctypes.byref(unique_id))
        data = torch.tensor(list(bytes(unique_id.internal)), dtype=torch.uint8)
        dist.broadcast(data, src=dist.get_process_group_ranks(group)[0], group=group)
        ctypes.memmove(ctypes.addressof(unique_id), bytes(data.tolist()), 128)
        self.comm = ctypes.c_void_p()
        with torch.cuda.device(device):
            self.nccl.call("ncclCommInitRank", ctypes.byref(self.comm), self.world_size, unique_id, self.rank)
            warmup = torch.zeros(1, device=device)
            self.all_reduce(warmup)
            torch.cuda.current_stream(device).synchronize()
        self.disabled = False

    def _check(self, tensor):
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError("PyNccl tensor 必须位于 communicator 设备且连续")

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        self._check(input_)
        output = torch.empty_like(input_)
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self.nccl.call("ncclAllReduce", input_.data_ptr(), output.data_ptr(), input_.numel(),
                       nccl_dtype(input_.dtype), 0, self.comm, stream)
        return output

    def all_gather(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        self._check(input_)
        self._check(output)
        if output.dtype != input_.dtype or output.numel() != input_.numel() * self.world_size:
            raise ValueError("all_gather 输出 tensor 的 dtype 或尺寸不正确")
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self.nccl.call("ncclAllGather", input_.data_ptr(), output.data_ptr(), input_.numel(),
                       nccl_dtype(input_.dtype), self.comm, stream)

    def destroy(self):
        if getattr(self, "disabled", True):
            return
        self.disabled = True
        # Like the frozen implementation, bound teardown even if another rank failed.
        def abort():
            with torch.cuda.device(self.device):
                self.nccl.call("ncclCommAbort", self.comm)
        thread = threading.Thread(target=abort, daemon=True)
        thread.start()
        thread.join(timeout=5.0)
