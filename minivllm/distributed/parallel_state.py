"""单机稠密 TP/PP 进程组；对应冻结 vLLM distributed.parallel_state。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist

_TP = None
_PP = None
_LOCAL_RANK = None
_DEVICE_INDEX = None
_OWNS_DEFAULT_GROUP = False


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


class GroupCoordinator:
    """一组成员的 CUDA/Gloo 进程组；全体 rank 必须按同序创建各子组。"""

    def __init__(self, group_ranks, local_rank: int, timeout: timedelta, device_index: int | None = None):
        self.rank = dist.get_rank()
        self.local_rank = local_rank
        self.device_index = local_rank if device_index is None else device_index
        self.device = torch.device('cuda', self.device_index)
        self.device_group = None
        self.cpu_group = None
        self._destroyed = False
        self.device_communicator = None
        owned = []
        try:
            for ranks in group_ranks:
                dg = dist.new_group(ranks, backend='nccl', timeout=timeout)
                if self.rank in ranks:
                    owned.append(dg)
                cg = dist.new_group(ranks, backend='gloo', timeout=timeout)
                if self.rank in ranks:
                    owned.append(cg)
                    self.ranks = list(ranks)
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_group, self.cpu_group = dg, cg
            if self.device_group is None or self.cpu_group is None:
                raise ValueError('当前 rank 不属于任何子组')
            if self.world_size > 1:
                from .device_communicators.cuda_communicator import CudaCommunicator
                self.device_communicator = CudaCommunicator(self.cpu_group, self.device)
        except BaseException:
            for group in reversed(owned):
                dist.destroy_process_group(group)
            raise

    def _check(self, tensor: torch.Tensor) -> None:
        if self._destroyed:
            raise RuntimeError('进程组已销毁')
        if tensor.device != self.device:
            raise ValueError(f'张量必须位于 {self.device}，收到 {tensor.device}')

    def _peer(self, rank: int) -> int:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError(f'组内 rank {rank!r} 越界')
        return self.ranks[rank]

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        self._check(input_)
        if self.world_size == 1:
            return input_
        return self.device_communicator.all_reduce(input_.contiguous())

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self._check(input_)
        if self.world_size == 1:
            return input_
        if not -input_.ndim <= dim < input_.ndim:
            raise ValueError(f'维度 {dim} 不适用于 {tuple(input_.shape)}')
        dim %= input_.ndim
        size = input_.size()
        output = torch.empty((size[0] * self.world_size,) + size[1:], dtype=input_.dtype, device=input_.device)
        self.device_communicator.all_gather(output, input_.contiguous())
        output = output.reshape((self.world_size,) + size).movedim(0, dim)
        return output.reshape(size[:dim] + (self.world_size * size[dim],) + size[dim + 1:])

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self._check(input_)
        if self.world_size == 1:
            return input_
        if not -input_.ndim <= dim < input_.ndim:
            raise ValueError(f'维度 {dim} 不适用于 {tuple(input_.shape)}')
        dim %= input_.ndim
        packed = input_.movedim(0, dim).contiguous()
        if packed.shape[0] % self.world_size:
            raise ValueError('待分片维度不能被组大小整除')
        output = torch.empty((packed.shape[0] // self.world_size,) + packed.shape[1:],
                             dtype=input_.dtype, device=input_.device)
        dist.reduce_scatter_tensor(output, packed, group=self.device_group)
        return output.movedim(0, dim).contiguous()

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        self._check(input_)
        peer = self._peer(src)
        if self.world_size > 1:
            dist.broadcast(input_, src=peer, group=self.device_group)
        return input_

    def broadcast_object(self, obj=None, src: int = 0):
        peer = self._peer(src)
        if self._destroyed:
            raise RuntimeError('进程组已销毁')
        if self.world_size == 1:
            return obj
        values = [obj]
        dist.broadcast_object_list(values, src=peer, group=self.cpu_group)
        return values[0]

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        self._check(tensor)
        if self.world_size == 1:
            raise ValueError('单 rank 组没有 device communicator')
        dst = (self.rank_in_group + 1) % self.world_size if dst is None else dst
        dist.send(tensor, dst=self._peer(dst), group=self.device_group)

    def recv(self, size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor:
        if self._destroyed:
            raise RuntimeError('进程组已销毁')
        if self.world_size == 1:
            raise ValueError('单 rank 组没有 device communicator')
        src = (self.rank_in_group - 1) % self.world_size if src is None else src
        output = torch.empty(size, dtype=dtype, device=self.device)
        dist.recv(output, src=self._peer(src), group=self.device_group)
        return output

    @contextmanager
    def graph_capture(self, graph_capture_context: GraphCaptureContext | None = None):
        """Capture on a shared explicit stream after initialization has completed."""
        if self._destroyed:
            raise RuntimeError('进程组已销毁')
        with torch.cuda.device(self.device):
            context = graph_capture_context or GraphCaptureContext(torch.cuda.Stream(device=self.device))
            current = torch.cuda.current_stream(self.device)
            if current != context.stream:
                context.stream.wait_stream(current)
            with torch.cuda.stream(context.stream):
                yield context

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self.device_communicator is not None:
            self.device_communicator.destroy()
            self.device_communicator = None
        for name in ('device_group', 'cpu_group'):
            group = getattr(self, name)
            if group is not None:
                dist.destroy_process_group(group)
                setattr(self, name, None)


def init_distributed_environment(world_size: int = -1, rank: int = -1,
                                 distributed_init_method: str = 'env://',
                                 local_rank: int = -1, backend: str = 'nccl',
                                 timeout: timedelta | None = None,
                                 device_index: int | None = None) -> None:
    global _LOCAL_RANK, _DEVICE_INDEX, _OWNS_DEFAULT_GROUP
    if backend != 'nccl':
        raise NotImplementedError('本专题设备组仅支持 NCCL')
    if local_rank == -1:
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
    if type(local_rank) is not int or local_rank < 0:
        raise ValueError(f'local_rank {local_rank!r} 必须是非负整数')
    if world_size > 0 and local_rank >= world_size:
        raise ValueError(f'local_rank {local_rank} 超出 world_size {world_size}')
    if device_index is None:
        device_index = local_rank
    if type(device_index) is not int or not 0 <= device_index < torch.cuda.device_count():
        raise ValueError(f'device_index {device_index!r} 超出可见 GPU')
    if _LOCAL_RANK is not None:
        raise RuntimeError('分布式环境已经初始化')
    torch.cuda.set_device(device_index)
    if dist.is_initialized():
        if (world_size != -1 and dist.get_world_size() != world_size) or (rank != -1 and dist.get_rank() != rank):
            raise ValueError('已存在的默认组与 rank/world_size 不匹配')
        if dist.get_backend() != 'nccl':
            raise ValueError('已存在的默认组不是 NCCL')
        owned = False
    else:
        dist.init_process_group(backend=backend, init_method=distributed_init_method,
                                world_size=world_size, rank=rank, timeout=timeout)
        owned = True
    _LOCAL_RANK, _DEVICE_INDEX, _OWNS_DEFAULT_GROUP = local_rank, device_index, owned


def initialize_model_parallel(tensor_model_parallel_size: int = 1,
                              pipeline_model_parallel_size: int = 1,
                              timeout: timedelta | None = None) -> None:
    global _TP, _PP
    if _LOCAL_RANK is None or not dist.is_initialized():
        raise RuntimeError('先初始化分布式环境')
    if _TP is not None or _PP is not None:
        raise RuntimeError('模型并行组已经初始化')
    tp, pp = tensor_model_parallel_size, pipeline_model_parallel_size
    if any(type(x) is not int or x < 1 for x in (tp, pp)) or tp * pp != dist.get_world_size():
        raise ValueError('TP×PP 必须等于单副本 world_size')
    timeout = timeout or timedelta(seconds=120)
    ranks = torch.arange(dist.get_world_size()).reshape(pp, tp)
    tp_groups = [row.tolist() for row in ranks]
    pp_groups = [column.tolist() for column in ranks.transpose(0, 1)]
    try:
        _TP = GroupCoordinator(tp_groups, _LOCAL_RANK, timeout, _DEVICE_INDEX)
        _PP = GroupCoordinator(pp_groups, _LOCAL_RANK, timeout, _DEVICE_INDEX)
    except BaseException:
        if _TP is not None:
            _TP.destroy()
            _TP = None
        raise


def get_tp_group_if_initialized() -> GroupCoordinator | None:
    """返回本进程的 TP 组；无分布式环境的既有单卡调用可检查 None。"""
    return _TP


def get_tp_group() -> GroupCoordinator:
    if _TP is None:
        raise RuntimeError('TP 组尚未初始化')
    return _TP


def get_pp_group() -> GroupCoordinator:
    if _PP is None:
        raise RuntimeError('PP 组尚未初始化')
    return _PP


def get_pp_group_if_initialized() -> GroupCoordinator | None:
    """保留未初始化分布式环境时的单阶段模型路径。"""
    return _PP


def destroy_model_parallel() -> None:
    global _TP, _PP
    for group in (_TP, _PP):
        if group is not None:
            group.destroy()
    _TP = _PP = None


def destroy_distributed_environment() -> None:
    global _LOCAL_RANK, _DEVICE_INDEX, _OWNS_DEFAULT_GROUP
    if _TP is not None or _PP is not None:
        raise RuntimeError('销毁默认组前先销毁模型并行组')
    if _OWNS_DEFAULT_GROUP and dist.is_initialized():
        dist.destroy_process_group()
    _LOCAL_RANK, _DEVICE_INDEX, _OWNS_DEFAULT_GROUP = None, None, False
