"""093 无模型部署探针；只在测试中替代模型加载阶段。"""
import os
import time
from pathlib import Path

from ..worker.worker import Worker


class ProbeWorker(Worker):
    def load_model(self):
        self._check_open()


class FailInitWorker(ProbeWorker):
    def init_device(self):
        if self.rank == 1:
            raise RuntimeError('injected init_device failure')
        super().init_device()


class FailLoadWorker(ProbeWorker):
    def load_model(self):
        if self.rank == 1:
            raise RuntimeError('injected load_model failure')
        super().load_model()


class DelayedWorker(ProbeWorker):
    def load_model(self):
        if self.rank == 1:
            root = Path(os.environ['MINIVLLM_STEP093_GATE'])
            (root / 'waiting').touch()
            deadline = time.monotonic() + 60
            while not (root / 'release').exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('测试 gate 等待超时')
                time.sleep(0.02)
        super().load_model()
