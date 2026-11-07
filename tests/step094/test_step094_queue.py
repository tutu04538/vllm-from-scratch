"""真实跨进程共享内存队列及冻结上游互操作。"""
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import threading
import time

import pytest
import torch
import zmq
from minivllm.distributed.shm_broadcast import MessageQueue

ROOT=Path(__file__).resolve().parents[2]


def step094_queue_reader(handle, rank, status, gate, use_upstream):
    if use_upstream:
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue as Queue
    else:
        Queue=MessageQueue
    queue=Queue.create_from_handle(handle,rank)
    try:
        queue.wait_until_ready()
        status.send('ready')
        if gate is not None:
            assert gate.wait(20)
        messages=[queue.dequeue(timeout=20) for _ in range(3)]
        status.send((messages[0],messages[1],messages[2].shape,messages[2].sum().item()))
    finally:
        if not use_upstream: queue.close()
        status.close()


@pytest.mark.parametrize('upstream_writer,upstream_reader',[(False,False),(False,True),(True,False)],
                         ids=['local','frozen_upstream_reader','frozen_upstream_writer'])
def test_cross_process_broadcast_overflow_and_slot_reuse(tmp_path,upstream_writer,upstream_reader):
    if upstream_writer or upstream_reader:
        import vllm.distributed.device_communicators.shm_broadcast as upstream
        frozen=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0/distributed/device_communicators/shm_broadcast.py'
        assert hashlib.sha256(frozen.read_bytes()).digest()==hashlib.sha256(Path(upstream.__file__).read_bytes()).digest()
    ctx=mp.get_context('spawn'); gate=ctx.Event()
    queue=(upstream.MessageQueue(2,2,max_chunk_bytes=512,max_chunks=1) if upstream_writer
           else MessageQueue(2,2,max_chunk_bytes=512,max_chunks=1,ipc_dir=str(tmp_path)))
    readers=[]; pipes=[]
    try:
        for rank in range(2):
            recv,send=ctx.Pipe(False)
            proc=ctx.Process(target=step094_queue_reader,args=(queue.export_handle(),rank,send,gate,upstream_reader))
            proc.start();send.close();readers.append(proc);pipes.append(recv)
        if upstream_writer:
            queue.local_socket.setsockopt(zmq.RCVTIMEO,30000)
            queue.wait_until_ready()
        else:
            queue.wait_until_ready(timeout=30)
        assert [p.recv() for p in pipes]==['ready','ready']
        queue.enqueue({'value':7},timeout=10)
        # 一槽尚未被两名读者消费时，第二条消息必须等待。
        release=threading.Timer(0.5,gate.set); release.start()
        start=time.monotonic(); queue.enqueue(b'overflow'*300,timeout=10); elapsed=time.monotonic()-start
        release.join(); assert elapsed>=0.25
        tensor=torch.arange(270000,dtype=torch.float32)
        queue.enqueue(tensor,timeout=20)
        for pipe in pipes:
            small,large,shape,total=pipe.recv()
            assert small=={'value':7} and large==b'overflow'*300
            assert tuple(shape)==(270000,) and total==tensor.sum().item()
        for proc in readers:
            proc.join(20); assert proc.exitcode==0
    finally:
        gate.set()
        for proc in readers:
            if proc.is_alive(): proc.terminate()
            proc.join(5)
        for pipe in pipes:pipe.close()
        if upstream_writer:
            handle=queue.export_handle()
            ctx_zmq=queue.local_socket.context
            queue.local_socket.close(linger=0)
            queue._spin_condition.local_notify_socket.close(linger=0)
            ctx_zmq.term()
            shm=queue.buffer.shared_memory
            del queue.buffer.shared_memory
            shm.close();shm.unlink()
            for address in (handle.local_subscribe_addr,handle.local_notify_addr):
                if address and address.startswith('ipc://'):
                    Path(address.removeprefix('ipc://')).unlink(missing_ok=True)
        else:
            queue.close()
    assert not list(tmp_path.iterdir())


def test_handshake_timeout_releases_owned_resources(tmp_path):
    queue=MessageQueue(1,1,max_chunk_bytes=512,max_chunks=1,ipc_dir=str(tmp_path))
    try:
        with pytest.raises(TimeoutError): queue.wait_until_ready(timeout=0.05)
    finally: queue.close()
    assert not list(tmp_path.iterdir())


def test_frozen_source_hashes_and_nonoutput_failure():
    from types import SimpleNamespace
    import vllm.v1.executor.multiproc_executor as upstream
    base=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'
    entries={e['path']:e['sha256'] for e in json.loads((base/'源码清单.json').read_text())['files']}
    installed=Path(importlib.util.find_spec('vllm').origin).parent
    for name in ('v1/executor/multiproc_executor.py','distributed/device_communicators/shm_broadcast.py',
                 'v1/serial_utils.py','v1/outputs.py','v1/worker/worker_base.py'):
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest(base/'vllm_0_28_0'/name)==entries[name]==digest(installed/name)
    class Inbox:
        n=0
        def dequeue(self,**kwargs):
            self.n+=1
            if self.n==1:return ('execute_model',(),{},0)
            raise SystemExit
    class Worker:
        def execute_model(self):raise RuntimeError('nonoutput failure')
    responses=[]
    obj=SimpleNamespace(rpc_broadcast_mq=Inbox(),worker=Worker(),rank=1,handle_output=responses.append)
    with pytest.raises(SystemExit):upstream.WorkerProc.worker_busy_loop(obj)
    assert responses==[]


def step094_inline_tensor_reader(handle, status):
    queue=MessageQueue.create_from_handle(handle,0)
    try:
        queue.wait_until_ready(timeout=20)
        first=queue.dequeue(timeout=20)
        expected=first.sum().item()
        status.send('first_received')
        second=queue.dequeue(timeout=20)
        status.send((first.sum().item(),expected,second.sum().item()))
    finally:
        queue.close();status.close()


def test_inline_oob_tensor_survives_slot_reuse(tmp_path):
    ctx=mp.get_context('spawn');recv,send=ctx.Pipe(False)
    queue=MessageQueue(1,1,max_chunk_bytes=2*1024*1024,max_chunks=1,ipc_dir=str(tmp_path))
    proc=ctx.Process(target=step094_inline_tensor_reader,args=(queue.export_handle(),send))
    try:
        proc.start();send.close();queue.wait_until_ready(timeout=30)
        tensor=torch.arange(270000,dtype=torch.float32)
        queue.enqueue(tensor,timeout=20)
        assert recv.recv()=='first_received'
        queue.enqueue(torch.zeros_like(tensor),timeout=20)
        after,before,second=recv.recv()
        assert before==after==tensor.sum().item() and second==0
        proc.join(20);assert proc.exitcode==0
    finally:
        if proc.is_alive():proc.terminate()
        proc.join(5);recv.close();queue.close()
    assert not list(tmp_path.iterdir())


def test_cuda_tensor_cannot_enter_cpu_serialization():
    import subprocess
    import sys
    script='''
import torch
from minivllm.distributed.shm_broadcast import _reduce_tensor
x=torch.ones(1,device='cuda')
try:_reduce_tensor(x)
except ValueError as e: assert 'CPU tensor' in str(e)
else:raise AssertionError('CUDA tensor must not be serialized')
'''
    p=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
