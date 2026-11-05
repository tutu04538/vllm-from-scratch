from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'benchmarks/check_parallel_environment.py'


def test_help_lists_communication():
    result = subprocess.run([sys.executable, str(ENTRY), '--help'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert '--communication' in result.stdout


def test_invalid_group_rejected(tmp_path):
    result = subprocess.run([sys.executable, str(ENTRY), '--communication', '--output', str(tmp_path),
                             '--groups', '0,0'], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'invalid GPU group' in result.stderr


def test_real_two_rank_collectives(tmp_path):
    result = subprocess.run([sys.executable, str(ENTRY), '--communication', '--output', str(tmp_path),
                             '--groups', '0,1', '--sizes', '4096', '--warmup', '1', '--iterations', '1',
                             '--repeats', '1', '--timeout', '120'], capture_output=True, text=True, timeout=140)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary[0]['exit_code'] == 0
    for rank in summary[0]['ranks']:
        assert rank['complete']
        assert {x['op'] for x in rank['measurements']} == {'all_reduce', 'all_gather', 'reduce_scatter', 'broadcast', 'send_recv'}
        assert all(x['correct'] and x['event_ms_mean'] > 0 for x in rank['measurements'])


def test_missing_rank_has_bounded_failure(tmp_path):
    env = os.environ.copy()
    env.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='2', MASTER_ADDR='127.0.0.1', MASTER_PORT='29691')
    worker = ROOT / 'benchmarks/step089_communication.py'
    result = subprocess.run([sys.executable, str(worker), '--worker', '--output', str(tmp_path), '--timeout', '2'],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    status = json.loads((tmp_path / 'rank0.json').read_text())
    assert status['rank'] == 0 and not status['complete'] and status['last'] == 'initializing'


def test_frozen_communicator_files_match_installed_vllm():
    import hashlib
    import importlib.util
    import ast

    base = ROOT / 'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'
    manifest = json.loads((base / '源码清单.json').read_text())
    files = {row['path']: row for row in manifest['files']}
    installed = Path(importlib.util.find_spec('vllm').origin).parent
    methods = {'distributed/device_communicators/cuda_communicator.py': ('CudaCommunicator', '__init__'),
               'distributed/device_communicators/custom_all_reduce.py': ('CustomAllreduce', '__init__'),
               'distributed/device_communicators/pynccl.py': ('PyNcclCommunicator', 'all_reduce')}
    for name, (cls, method) in methods.items():
        snapshot = (base / 'vllm_0_28_0' / name).read_bytes()
        assert hashlib.sha256(snapshot).hexdigest() == files[name]['sha256']
        assert hashlib.sha256((installed / name).read_bytes()).hexdigest() == files[name]['sha256']
        tree = ast.parse(snapshot)
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        assert any(isinstance(n, ast.FunctionDef) and n.name == method for n in node.body)


def test_worker_failure_propagates_to_parent(tmp_path):
    result = subprocess.run([sys.executable, str(ENTRY), '--communication', '--output', str(tmp_path),
                             '--groups', '0,99', '--sizes', '4096', '--warmup', '1', '--iterations', '1',
                             '--repeats', '1', '--timeout', '30'], capture_output=True, text=True, timeout=45)
    assert result.returncode != 0
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary[0]['exit_code'] != 0
    assert 'rank' in result.stderr


def test_parent_timeout_is_bounded(tmp_path):
    result = subprocess.run([sys.executable, str(ENTRY), '--communication', '--output', str(tmp_path),
                             '--groups', '0,1', '--timeout', '1'], capture_output=True, text=True, timeout=20)
    assert result.returncode == 124
    assert json.loads((tmp_path / 'summary.json').read_text())[0]['exit_code'] == 124
    assert 'rank' in result.stderr
