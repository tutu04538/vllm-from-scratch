"""Exercise the actual hash-matched frozen vLLM worker-monitor method."""
import hashlib
import multiprocessing as mp
from pathlib import Path
import threading
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0/v1/executor/multiproc_executor.py'


def exit_worker():
    import os
    os._exit(23)


def test_frozen_upstream_monitor():
    from vllm.v1.executor import multiproc_executor as upstream
    assert hashlib.sha256(Path(upstream.__file__).read_bytes()).digest() == hashlib.sha256(FROZEN.read_bytes()).digest()
    proc = mp.get_context('spawn').Process(target=exit_worker)
    proc.start()
    callback = threading.Event()
    shutdown = threading.Event()
    class Harness:
        workers = [SimpleNamespace(proc=proc)]
        shutting_down = False
        is_failed = False
        failure_callback = callback.set
        def shutdown(self):
            shutdown.set()
    harness = Harness()
    try:
        upstream.MultiprocExecutor.start_worker_monitor(harness)
        assert callback.wait(20)
        assert shutdown.is_set() and harness.is_failed
        assert harness.failure_callback is None
    finally:
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(10)
    assert proc.exitcode == 23
