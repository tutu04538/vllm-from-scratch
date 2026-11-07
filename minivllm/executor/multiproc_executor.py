"""单机 Worker 生命周期；正式推理 RPC 和模型分片尚未开放。"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import pickle
import tempfile
import time
import traceback

from ..worker.worker import Worker


def _check_config_for_cuda(value, seen=None):
    """配置可以序列化，但不能把父进程的 CUDA 张量作为启动参数。"""
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            raise ValueError('Worker 配置不能包含 CUDA 张量')
        return
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if is_dataclass(value) and not isinstance(value, type):
        children = (getattr(value, f.name) for f in fields(value))
    elif isinstance(value, dict):
        children = (*value.keys(), *value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    else:
        return
    for child in children:
        _check_config_for_cuda(child, seen)


@dataclass
class UnreadyWorkerProcHandle:
    rank: int
    process: mp.Process
    ready_pipe: object
    death_writer: object


class WorkerProc:
    """子进程入口与 CPU readiness 管道，对照上游 WorkerProc。"""

    @staticmethod
    def make_worker_process(context, vllm_config, rank, device_index, init_method, worker_cls):
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        proc = context.Process(target=WorkerProc.worker_main,
            args=(vllm_config, rank, device_index, init_method, worker_cls, ready_writer, death_reader),
            name=f'MiniVllmWorker-{rank}', daemon=True)
        try:
            proc.start()
        except BaseException:
            ready_reader.close(); ready_writer.close(); death_reader.close(); death_writer.close()
            raise
        ready_writer.close()
        death_reader.close()
        return UnreadyWorkerProcHandle(rank, proc, ready_reader, death_writer)

    @staticmethod
    def worker_main(vllm_config, rank, device_index, init_method, worker_cls, ready, death):
        worker = None
        phase = 'construct_worker'
        try:
            worker = worker_cls(vllm_config, rank=rank, local_rank=rank,
                                device_index=device_index, distributed_init_method=init_method)
            phase = 'init_device'
            worker.init_device()
            phase = 'load_model'
            worker.load_model()
            import torch
            from ..distributed import get_tp_group, get_pp_group
            props = torch.cuda.get_device_properties(device_index)
            ready.send({'status':'READY', 'rank':rank, 'local_rank':rank, 'pid':os.getpid(),
                        'device_index':device_index, 'device_uuid':str(props.uuid),
                        'tp_ranks':get_tp_group().ranks, 'pp_ranks':get_pp_group().ranks,
                        'group_local_rank':get_tp_group().local_rank,
                        'group_device_index':get_tp_group().device_index,
                        'model_loaded':worker.model_runner is not None})
            ready.close()
            ready = None
            phase = 'wait_for_shutdown'
            try:
                death.recv_bytes()
            except EOFError:
                pass
        except BaseException as exc:
            if ready is not None:
                try:
                    ready.send({'status':'ERROR', 'rank':rank, 'phase':phase,
                                'error':f'{type(exc).__name__}: {exc}', 'traceback':traceback.format_exc()})
                except (BrokenPipeError, EOFError, OSError):
                    pass
        finally:
            if ready is not None:
                ready.close()
            death.close()
            if worker is not None:
                try:
                    worker.shutdown()
                except BaseException:
                    traceback.print_exc()

    @staticmethod
    def wait_for_ready(handles, timeout):
        deadline = time.monotonic() + timeout
        pending = {h.ready_pipe:h for h in handles}
        results = {}
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'等待 Worker ready 超时，未就绪 ranks={[h.rank for h in pending.values()]}')
            events = wait(list(pending), timeout=remaining)
            if not events:
                raise TimeoutError(f'等待 Worker ready 超时，未就绪 ranks={[h.rank for h in pending.values()]}')
            for pipe in events:
                handle = pending.pop(pipe)
                try:
                    message = pipe.recv()
                except EOFError as exc:
                    raise RuntimeError(f'Worker rank {handle.rank} 初始化时退出，未返回 ready') from exc
                finally:
                    pipe.close()
                if message.get('status') != 'READY':
                    raise RuntimeError(f"Worker rank {handle.rank} 在 {message.get('phase')} 失败: "
                                       f"{message.get('error')}\n{message.get('traceback', '')}")
                if message.get('rank') != handle.rank:
                    raise RuntimeError(f'Worker rank {handle.rank} 的 ready 身份不匹配')
                results[handle.rank] = message
        return [results[h.rank] for h in handles]


class MultiprocExecutor:
    """拥有 Worker 启停；execute/采样和多卡模型加载分别留待后续关卡。"""

    def __init__(self, vllm_config, *, _worker_cls=None, _startup_timeout=120):
        p = vllm_config.parallel_config
        vllm_config._validate_parallel_config()
        if p.world_size < 2 or p.data_parallel_size != 1 or p.resolved_backend != 'mp':
            raise ValueError('MultiprocExecutor 要求单副本、world_size>1、mp 后端')
        if type(_startup_timeout) not in (int, float) or _startup_timeout <= 0:
            raise ValueError('启动超时必须为正')
        _check_config_for_cuda(vllm_config)
        pickle.dumps(vllm_config)
        self.vllm_config = vllm_config
        self.workers = []
        self.ready_workers = []
        self._closed = False
        self._rendezvous = tempfile.TemporaryDirectory(prefix='minivllm-mp-')
        init_method = (Path(self._rendezvous.name) / 'dist.store').as_uri()
        context = mp.get_context('spawn')
        worker_cls = _worker_cls or Worker
        try:
            for rank in range(p.world_size):
                device_index = p.device_ids[rank] if p.device_ids is not None else rank
                self.workers.append(WorkerProc.make_worker_process(
                    context, vllm_config, rank, device_index, init_method, worker_cls))
            self.ready_workers = WorkerProc.wait_for_ready(self.workers, _startup_timeout)
            for handle in self.workers:
                if not handle.process.is_alive():
                    raise RuntimeError(f'Worker rank {handle.rank} 在 ready 后立即退出')
        except BaseException as exc:
            try:
                self.shutdown()
            except BaseException as cleanup_error:
                raise RuntimeError(f'{exc}; Worker 清理失败: {cleanup_error}') from exc
            raise

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        for handle in self.workers:
            handle.death_writer.close()
            handle.ready_pipe.close()
        deadline = time.monotonic() + 5
        for handle in self.workers:
            handle.process.join(max(0, deadline - time.monotonic()))
        for handle in self.workers:
            if handle.process.is_alive():
                handle.process.terminate()
        deadline = time.monotonic() + 5
        for handle in self.workers:
            handle.process.join(max(0, deadline - time.monotonic()))
        for handle in self.workers:
            if handle.process.is_alive():
                handle.process.kill()
        deadline = time.monotonic() + 5
        for handle in self.workers:
            handle.process.join(max(0, deadline - time.monotonic()))
        remaining = [h.rank for h in self.workers if h.process.is_alive()]
        self._rendezvous.cleanup()
        if remaining:
            raise RuntimeError(f'Worker 终止超时，未退出 ranks={remaining}')

    def get_cache_config(self):
        raise NotImplementedError('多卡模型 KV 和执行 RPC 尚未开放')

    def execute_model(self, *args, **kwargs):
        raise NotImplementedError('多卡模型执行 RPC 尚未开放')
