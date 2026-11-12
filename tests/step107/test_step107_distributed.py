"""Real NCCL smoke checks; full model and frozen differential require separate acceptance."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize('world', [2, 4])
def test_local_heads_and_global_mapping(world):
    if torch.cuda.device_count() < world:
        pytest.skip(f'requires {world} CUDA devices')
    worker = Path(__file__).with_name('step107_rank_worker.py')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                             f'--nproc-per-node={world}', str(worker)],
                            env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert result.stdout.count('passed') == world
