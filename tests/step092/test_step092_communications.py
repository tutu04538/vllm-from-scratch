"""真实多进程 GPU 通信，单个用例自行启动，不依赖预跑 benchmark。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
ENTRY=ROOT/'benchmarks/check_step092_acceptance.py'
spec=importlib.util.spec_from_file_location('step092_acceptance',ENTRY)
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


@pytest.mark.parametrize('case',[(1,1,1,None,False),(2,2,1,None,False),
                                (4,4,1,None,False),(4,2,2,None,False),
                                (2,2,1,None,True),(2,2,1,1,False)],
                         ids=['one','two','four','mixed','frozen_oracle','partial_failure'])
def test_real_gpu_communications(case,tmp_path):
    result=runner.execute(case,tmp_path,120)
    log=Path(result['log']).read_text()
    assert result['passed'], log[-12000:]
    if case[3] is None:
        for rank in result['ranks']:
            assert rank['complete'] and len(rank['cycles'])==2
            for cycle in rank['cycles']:
                assert cycle['uuid'] and cycle['tp_rank'] == cycle['tp_ranks'].index(rank['rank'])
                assert cycle['pp_rank'] == cycle['pp_ranks'].index(rank['rank'])
        if case[0:3] == (4,2,2):
            assert [rank['cycles'][0]['all_reduce'][0] for rank in result['ranks']]==[3,3,7,7]
            assert [rank['cycles'][0]['pp_all_reduce'][0] for rank in result['ranks']]==[4,6,4,6]
    else:
        assert result['exit_code'] not in (0,124)
        assert 'stage=partial-group-created' in log
        assert 'injected rank failure after partial group creation' in log


def test_help_and_bad_timeout(tmp_path):
    for args in (['--help'], ['--output',str(tmp_path/'x.json'),'--timeout','0']):
        p=subprocess.run([sys.executable,str(ENTRY),*args],capture_output=True,text=True,timeout=20)
        assert (p.returncode==0) == (args==['--help'])


def test_frozen_source_hashes():
    ref=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'
    manifest=json.loads((ref/'源码清单.json').read_text())
    entries={e['path']:e['sha256'] for e in manifest['files']}
    installed=Path(importlib.util.find_spec('vllm').origin).parent
    for name in ('distributed/parallel_state.py','distributed/communication_op.py',
                 'distributed/device_communicators/base_device_communicator.py',
                 'distributed/device_communicators/cuda_communicator.py',
                 'distributed/device_communicators/pynccl.py'):
        snap=(ref/'vllm_0_28_0'/name).read_bytes(); installed_bytes=(installed/name).read_bytes()
        assert hashlib.sha256(snap).hexdigest()==entries[name]==hashlib.sha256(installed_bytes).hexdigest()


def test_cuda_visible_devices_remapping(tmp_path, monkeypatch):
    query=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    uuids=[line.strip().removeprefix('GPU-') for line in query.stdout.splitlines()]
    assert len(uuids) >= 3
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,0')
    result=runner.execute((2,2,1,None,False),tmp_path,120)
    assert result['passed'], Path(result['log']).read_text()[-12000:]
    assert [r['cycles'][0]['uuid'] for r in result['ranks']]==[uuids[2],uuids[0]]
    assert [r['cycles'][0]['local_rank'] for r in result['ranks']]==[0,1]
