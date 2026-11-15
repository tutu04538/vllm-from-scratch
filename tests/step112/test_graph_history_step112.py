"""生产图管理器不会把每轮选择保存在生命周期历史中。"""
import os
from pathlib import Path
import subprocess
import sys


def test_graph_dispatch_does_not_accumulate_request_history():
    root = Path(__file__).resolve().parents[2]
    code = '''
import torch
from spec74_helpers import make_config
from minivllm.config import CUDAGraphMode
from minivllm.worker.gpu.cudagraph_utils import CudaGraphManager
from minivllm.worker.gpu_model_runner import GPUModelRunner
cfg = make_config(spec_k=3, mode="full_decode_only", capture_sizes=[1, 2, 4, 8])
manager = CudaGraphManager(cfg, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY,
                           decode_query_len=1, allow_shape_padding=True)
manager._graphs_captured = True
before = {k: repr(v) for k, v in vars(manager).items()}
for _ in range(10000):
    desc = manager.dispatch(2, 2, 1)
    assert desc.num_tokens == 2
assert {k: repr(v) for k, v in vars(manager).items()} == before
assert not hasattr(manager, "selections")
assert not hasattr(GPUModelRunner, "_record_selection")
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(root / 'tests/step74'))))
    run = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
