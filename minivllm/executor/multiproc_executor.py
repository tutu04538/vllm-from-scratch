"""单机多进程 Worker 与共享内存 RPC；模型分片尚未开放。"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from enum import Enum, auto
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import pickle
import tempfile
import time
import threading
import traceback
import uuid
from multiprocessing import shared_memory
from unittest.mock import patch
import cloudpickle

from ..worker.worker import Worker
from ..distributed.shm_broadcast import MessageQueue
from ..outputs import AsyncModelRunnerOutput


class FutureWrapper(Future):
    """按提交顺序读取队列；对应冻结上游 FutureWrapper。"""
    def __init__(self, futures_queue, get_response, aggregate=lambda x:x):
        super().__init__()
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
        self.futures_queue.appendleft(self)

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError('FutureWrapper.result timeout 尚未实现')
        while not self.done():
            self.futures_queue.pop()._wait_for_response()
        return super().result()

    def _wait_for_response(self):
        try:
            result = self.aggregate(self.get_response())
            with suppress(InvalidStateError):
                self.set_result(result)
        except Exception as exc:
            with suppress(InvalidStateError):
                self.set_exception(exc)


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
    failure_reader: object


class WorkerProc:
    """子进程入口与 CPU readiness 管道，对照上游 WorkerProc。"""

    @staticmethod
    def make_worker_process(context, vllm_config, rank, device_index, init_method, worker_cls,
                            input_handle, startup_timeout, ipc_dir, response_name):
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        failure_reader, failure_writer = context.Pipe(duplex=False)
        proc = context.Process(target=WorkerProc.worker_main,
            args=(vllm_config, rank, device_index, init_method, worker_cls, ready_writer, death_reader,
                  input_handle, startup_timeout, ipc_dir, response_name, failure_writer),
            name=f'MiniVllmWorker-{rank}', daemon=True)
        try:
            proc.start()
        except BaseException:
            ready_reader.close(); ready_writer.close(); death_reader.close(); death_writer.close()
            failure_reader.close(); failure_writer.close()
            raise
        ready_writer.close()
        death_reader.close()
        failure_writer.close()
        return UnreadyWorkerProcHandle(rank, proc, ready_reader, death_writer, failure_reader)

    @staticmethod
    def worker_main(vllm_config, rank, device_index, init_method, worker_cls, ready, death,
                    input_handle, startup_timeout, ipc_dir, response_name, failure_writer):
        worker = None
        input_mq = response_mq = None
        queue_lifecycle_lock = threading.Lock()
        phase = 'construct_worker'
        try:
            worker = worker_cls(vllm_config, rank=rank, local_rank=rank,
                                device_index=device_index, distributed_init_method=init_method)
            phase = 'init_device'
            worker.init_device()
            phase = 'load_model'
            worker.load_model()
            phase = 'init_message_queues'
            input_mq = MessageQueue.create_from_handle(input_handle, rank)
            response_mq = MessageQueue(1, 1, ipc_dir=ipc_dir, buffer_name=response_name)
            import torch
            from ..distributed import get_tp_group, get_pp_group
            props = torch.cuda.get_device_properties(device_index)
            ready.send({'status':'READY', 'rank':rank, 'local_rank':rank, 'pid':os.getpid(),
                        'device_index':device_index, 'device_uuid':str(props.uuid),
                        'tp_ranks':get_tp_group().ranks, 'pp_ranks':get_pp_group().ranks,
                        'group_local_rank':get_tp_group().local_rank,
                        'group_device_index':get_tp_group().device_index,
                        'model_loaded':worker.model_runner is not None,
                        'response_handle':response_mq.export_handle()})
            ready.close()
            ready = None
            input_mq.wait_until_ready(timeout=startup_timeout)
            response_mq.wait_until_ready(timeout=startup_timeout)
            phase = 'worker_busy_loop'
            shutdown_requested = threading.Event()
            def monitor_parent():
                try:
                    death.recv_bytes()
                except (EOFError, OSError):
                    shutdown_requested.set()
                    with queue_lifecycle_lock:
                        if not getattr(input_mq, '_closed', False):
                            input_mq.shutdown()
            threading.Thread(target=monitor_parent, daemon=True,
                             name='DeathPipeMonitor').start()
            WorkerProc.worker_busy_loop(worker, rank, input_mq, response_mq,
                                        shutdown_requested, failure_writer)
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
            failure_writer.close()
            with queue_lifecycle_lock:
                if input_mq is not None:
                    input_mq.close()
                if response_mq is not None:
                    response_mq.close()
            if worker is not None:
                try:
                    worker.shutdown()
                except BaseException:
                    traceback.print_exc()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    @staticmethod
    def worker_busy_loop(worker, rank, input_mq, response_mq, shutdown_requested, failure_writer):
        while not shutdown_requested.is_set():
            try:
                method, args, kwargs, output_rank = input_mq.dequeue(timeout=0.2, indefinite=True)
            except TimeoutError:
                continue
            except RuntimeError:
                if shutdown_requested.is_set():
                    return
                raise
            try:
                func = getattr(worker, method) if isinstance(method, str) else lambda *a, **k: cloudpickle.loads(method)(worker, *a, **k)
                output = func(*args, **kwargs)
                if isinstance(output, AsyncModelRunnerOutput):
                    output = output.get_output()
                # 同一 response queue 格式；非输出 rank 发空成功回执，使失败不能被忽略。
                response_mq.enqueue((WorkerProc.ResponseStatus.SUCCESS,
                                     output if output_rank is None or rank == output_rank else None))
            except Exception as exc:
                error = (f'rank {rank} pid {os.getpid()} 阶段 {method}: '
                         f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
                # 独立控制管道让 monitor 即使在等待较早 rank 的响应时也能发现原始异常。
                try:
                    failure_writer.send(error)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                response_mq.enqueue((WorkerProc.ResponseStatus.FAILURE, error))
                return

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
    """拥有 Worker 生命周期和 RPC；多卡模型参数分片尚未开放。"""

    def __init__(self, vllm_config, *, _worker_cls=None, _startup_timeout=120):
        p = vllm_config.parallel_config
        vllm_config._validate_parallel_config()
        if p.world_size < 2 or p.data_parallel_size != 1 or p.resolved_backend != 'mp':
            raise ValueError('MultiprocExecutor 要求单副本、world_size>1、mp 后端')
        if type(_startup_timeout) not in (int, float) or _startup_timeout <= 0:
            raise ValueError('启动超时必须为正')
        _check_config_for_cuda(vllm_config)
        # The normal production Worker can validate model/backend metadata in
        # the parent, before spawning workers or creating process groups. Test
        # workers deliberately exercising the RPC lifecycle may have no model.
        if _worker_cls is None:
            Worker(vllm_config)._validate_tp_model()
        pickle.dumps(vllm_config)
        self.vllm_config = vllm_config
        self.workers = []
        self.ready_workers = []
        self.rpc_broadcast_mq = None
        self.response_mqs = []
        self._response_shm_names = [f'minivllm094_{uuid.uuid4().hex}' for _ in range(p.world_size)]
        self.futures_queue = deque()
        self.failure = None
        self.failure_callback = None
        self._state_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._response_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._monitor_thread = None
        self.output_rank = p.world_size - p.tensor_parallel_size
        self._closed = False
        self._rendezvous = tempfile.TemporaryDirectory(prefix='minivllm-mp-')
        init_method = (Path(self._rendezvous.name) / 'dist.store').as_uri()
        context = mp.get_context('spawn')
        worker_cls = _worker_cls or Worker
        try:
            self.rpc_broadcast_mq = MessageQueue(p.world_size, p.world_size, ipc_dir=self._rendezvous.name)
            input_handle = self.rpc_broadcast_mq.export_handle()
            for rank in range(p.world_size):
                device_index = p.device_ids[rank] if p.device_ids is not None else rank
                self.workers.append(WorkerProc.make_worker_process(
                    context, vllm_config, rank, device_index, init_method, worker_cls,
                    input_handle, _startup_timeout, self._rendezvous.name,
                    self._response_shm_names[rank]))
            self.ready_workers = WorkerProc.wait_for_ready(self.workers, _startup_timeout)
            for handle in self.workers:
                if not handle.process.is_alive():
                    raise RuntimeError(f'Worker rank {handle.rank} 在 ready 后立即退出')
            for info in self.ready_workers:
                self.response_mqs.append(MessageQueue.create_from_handle(info.pop('response_handle'), 0))
            self.rpc_broadcast_mq.wait_until_ready(timeout=_startup_timeout)
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready(timeout=_startup_timeout)
            self.start_worker_monitor()
        except BaseException as exc:
            try:
                self.shutdown()
            except BaseException as cleanup_error:
                raise RuntimeError(f'{exc}; Worker 清理失败: {cleanup_error}') from exc
            raise

    def _mark_failure(self, message):
        callback = None
        with self._state_lock:
            if self.failure is None and not self._closed:
                self.failure = str(message)
                callback, self.failure_callback = self.failure_callback, None
        if callback is not None:
            try:
                callback()
            except Exception:
                traceback.print_exc()
        if self.failure is not None and not self._closed:
            threading.Thread(target=self.shutdown, daemon=True,
                             name='MultiprocFailureCleanup').start()

    def register_failure_callback(self, callback):
        with self._state_lock:
            failed = self.failure is not None
            if not failed:
                self.failure_callback = callback
        if failed:
            callback()

    def start_worker_monitor(self):
        """观察所有 Worker，不能仅等待当前 rank 的响应。"""
        def monitor():
            watched = {h.process.sentinel: h for h in self.workers}
            watched.update({h.failure_reader: h for h in self.workers})
            while watched:
                events = wait(list(watched))
                # 优先读取 Worker 发来的原始异常，避免被退出码覆盖。
                events.sort(key=lambda event: isinstance(event, int))
                for event in events:
                    handle = watched.get(event)
                    if handle is None:
                        continue
                    if event is handle.failure_reader:
                        try:
                            message = event.recv()
                        except (EOFError, OSError):
                            watched.pop(event, None)
                            continue
                    else:
                        handle.process.join(timeout=0.1)
                        message = (f'Worker rank {handle.rank} pid {handle.process.pid} '
                                   f'意外退出，exitcode={handle.process.exitcode}')
                    with self._state_lock:
                        closing = self._closed
                    if not closing:
                        self._mark_failure(message)
                        self.shutdown()
                    return

        self._monitor_thread = threading.Thread(target=monitor, daemon=True,
                                                name='MultiprocWorkerMonitor')
        self._monitor_thread.start()

    def shutdown(self):
        # monitor 和控制线程可能同时关闭；必须等首次清理完成，而非只检查标志后返回。
        with self._shutdown_lock:
            if self._closed:
                return
            with self._state_lock:
                self._closed = True
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.shutdown()
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
            # 等待读响应的线程离开 dequeue，再释放底层 socket / SHM。
            with self._response_lock:
                for response_mq in self.response_mqs:
                    response_mq.close()
                with self._send_lock:
                    if self.rpc_broadcast_mq is not None:
                        self.rpc_broadcast_mq.close()
            for handle in self.workers:
                handle.failure_reader.close()
            for name in self._response_shm_names:
                try:
                    with patch('multiprocessing.resource_tracker.register', lambda *args, **kwargs: None):
                        orphan = shared_memory.SharedMemory(name=name)
                except FileNotFoundError:
                    continue
                try:
                    with suppress(FileNotFoundError):
                        orphan.unlink()
                finally:
                    orphan.close()
            self._rendezvous.cleanup()
            if remaining:
                raise RuntimeError(f'Worker 终止超时，未退出 ranks={remaining}')

    def check_health(self):
        if self.failure is not None:
            raise RuntimeError(f'Executor 已失败: {self.failure}')
        self.collective_rpc('check_health', timeout=10)

    def get_cache_config(self):
        if any(not info['model_loaded'] for info in self.ready_workers):
            raise NotImplementedError('无模型 Worker 不能初始化 KV 配置')
        configs = self.collective_rpc('get_cache_config')
        from ..core.kv_cache_interface import KVCacheConfig
        if all(isinstance(config, KVCacheConfig) for config in configs):
            from ..core.kv_cache_utils import (get_kv_cache_configs,
                                               generate_scheduler_kv_cache_config)
            available = self.collective_rpc('determine_available_memory')
            self.worker_kv_cache_configs = get_kv_cache_configs(
                configs, available, self.vllm_config.model_config.max_model_len,
                reserve_null_block=self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs())
            self.worker_kv_available_memory = available
            return generate_scheduler_kv_cache_config(self.worker_kv_cache_configs)
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError('Worker KV 配置不一致，尚不支持聚合')
        return configs[0]

    def initialize_kv_cache(self, kv_cache_config):
        self.collective_rpc('initialize_from_config', args=(kv_cache_config,))
        self.collective_rpc('compile_or_warm_up_model')

    def supports_async_scheduling(self):
        return False

    def execute_model(self, scheduler_output, non_block=False):
        from ..core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
        if not isinstance(scheduler_output, SchedulerOutput):
            raise TypeError('execute_model 只接收 SchedulerOutput 快照')
        if (not isinstance(scheduler_output.scheduled_cached_reqs, CachedRequestData)
                or any(not isinstance(r, NewRequestData) for r in scheduler_output.scheduled_new_reqs)):
            raise TypeError('SchedulerOutput 必须包含请求数据快照，不能传递 Request 活对象')
        return self.collective_rpc('execute_model', args=(scheduler_output,),
                                   unique_reply_rank=self.output_rank, non_block=non_block)

    def sample_tokens(self, grammar_output, non_block=False):
        return self.collective_rpc('sample_tokens', args=(grammar_output,),
                                   unique_reply_rank=self.output_rank, non_block=non_block)

    def take_draft_token_ids(self):
        return self.collective_rpc('take_draft_token_ids', unique_reply_rank=self.output_rank)

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False,
                       unique_reply_rank=None):
        if self.failure is not None:
            raise RuntimeError(f'Executor 已失败: {self.failure}')
        if self._closed:
            raise RuntimeError('Executor 已关闭')
        if unique_reply_rank is not None and not 0 <= unique_reply_rank < len(self.workers):
            raise ValueError('unique_reply_rank 越界')
        deadline = None if timeout is None else time.monotonic() + timeout
        send_method = method if isinstance(method, str) else cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            with self._send_lock:
                if self.failure is not None:
                    raise RuntimeError(f'Executor 已失败: {self.failure}')
                if self._closed:
                    raise RuntimeError('Executor 已关闭')
                self.rpc_broadcast_mq.enqueue((send_method, args, kwargs or {}, unique_reply_rank), timeout=timeout)
        except Exception as exc:
            self._mark_failure(f'RPC {method} 发送失败: {exc}')
            raise

        def get_response():
            responses = []
            try:
                if self.failure is not None:
                    raise RuntimeError(f'Executor 已失败: {self.failure}')
                for rank, mq in enumerate(self.response_mqs):
                    while True:
                        remaining = None if deadline is None else deadline - time.monotonic()
                        if remaining is not None and remaining <= 0:
                            raise TimeoutError(f'RPC {method} 超时')
                        try:
                            with self._response_lock:
                                if self.failure is not None:
                                    raise RuntimeError(f'Executor 已失败: {self.failure}')
                                status, value = mq.dequeue(timeout=min(0.2, remaining) if remaining is not None else 0.2)
                            break
                        except TimeoutError:
                            if not self.workers[rank].process.is_alive():
                                raise RuntimeError(f'Worker rank {rank} 在 RPC {method} 中退出')
                    if status != WorkerProc.ResponseStatus.SUCCESS:
                        raise RuntimeError(f'Worker 失败: {value}')
                    responses.append(value)
            except Exception as exc:
                self._mark_failure(str(exc))
                raise
            if self.failure is not None:
                raise RuntimeError(f'Executor 已失败: {self.failure}')
            return responses[unique_reply_rank] if unique_reply_rank is not None else responses

        future = FutureWrapper(self.futures_queue, get_response)
        return future if non_block else future.result()
