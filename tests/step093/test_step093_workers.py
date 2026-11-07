"""093 真多进程设备所有权、启动失败和 readiness 门控。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'benchmarks/check_step093_acceptance.py'


@pytest.mark.parametrize('case', ['2','4','device-ids','fail-device','fail-load','production'])
def test_real_worker_lifecycle(case, tmp_path):
    output=tmp_path/f'{case}.json'
    p=subprocess.run([sys.executable,str(ENTRY),'--case',case,'--timeout','120','--output',str(output)],
                     cwd=ROOT, capture_output=True,text=True,timeout=150)
    assert p.returncode == 0, p.stdout+'\n'+p.stderr
    result=json.loads(output.read_text())
    assert result['status']=='passed'
    assert not result['parent_cuda_initialized_before'] and not result['parent_cuda_initialized_after']
    for attempt in result['cases'][0]['attempts']:
        assert attempt['children_reaped'] and attempt['rendezvous_cleaned']
        for worker in attempt.get('ready',[]):
            assert worker['pid'] != result['parent_pid']
            assert worker['tp_ranks'] == list(range(result['cases'][0]['workers']))
            assert not worker['model_loaded']


def test_device_remapping(tmp_path):
    output=tmp_path/'mapped.json'
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'2,0'}
    p=subprocess.run([sys.executable,str(ENTRY),'--case','2','--timeout','120','--output',str(output)],
                     cwd=ROOT,env=env,capture_output=True,text=True,timeout=150)
    assert p.returncode == 0,p.stdout+'\n'+p.stderr
    workers=json.loads(output.read_text())['cases'][0]['attempts'][0]['ready']
    assert [w['device_uuid'] for w in workers]==[uuids[2].removeprefix('GPU-'),uuids[0].removeprefix('GPU-')]


def test_no_scheduler_until_every_worker_ready(tmp_path, monkeypatch):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1')
    monkeypatch.setenv('MINIVLLM_STEP093_GATE',str(tmp_path))
    from minivllm.config import DeviceConfig, ModelConfig, ParallelConfig, VllmConfig
    from minivllm.engine import LLMEngine
    from minivllm.executor import MultiprocExecutor
    from minivllm.testing.step093_probe import DelayedWorker
    from minivllm.core.sched.scheduler import Scheduler
    cfg=VllmConfig(ModelConfig('unused',hf_config={'num_attention_heads':4,'num_key_value_heads':2,
                     'num_hidden_layers':2},enforce_eager=True),device_config=DeviceConfig('cuda'),
                   parallel_config=ParallelConfig(tensor_parallel_size=2))
    scheduled=threading.Event()
    original=Scheduler.__init__
    def spy(self,*args,**kwargs):
        scheduled.set()
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Scheduler,'__init__',spy)
    monkeypatch.setattr('minivllm.executor.create_executor',
                        lambda config: MultiprocExecutor(config,_worker_cls=DelayedWorker,_startup_timeout=60))
    result=[]
    def launch():
        try:
            LLMEngine(cfg)
        except Exception as exc:
            result.append(exc)
    thread=threading.Thread(target=launch,daemon=True); thread.start()
    try:
        deadline=time.monotonic()+45
        while not (tmp_path/'waiting').exists() and time.monotonic()<deadline:
            if not thread.is_alive(): break
            time.sleep(0.02)
        assert (tmp_path/'waiting').exists(), result
        assert thread.is_alive() and not scheduled.is_set() and not result
    finally:
        (tmp_path/'release').touch()
        thread.join(timeout=60)
    assert not thread.is_alive()
    assert len(result)==1 and isinstance(result[0],NotImplementedError)
    assert 'KV' in str(result[0]) and not scheduled.is_set()


def test_frozen_source_matches_installed():
    base=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'
    manifest=json.loads((base/'源码清单.json').read_text())
    entries={e['path']:e['sha256'] for e in manifest['files']}
    installed=Path(importlib.util.find_spec('vllm').origin).parent
    for name in ('v1/executor/multiproc_executor.py','v1/worker/worker_base.py',
                 'v1/worker/gpu_worker.py','utils/system_utils.py'):
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest(base/'vllm_0_28_0'/name)==entries[name]==digest(installed/name)


def test_help_and_invalid_timeout(tmp_path):
    help_result=subprocess.run([sys.executable,str(ENTRY),'--help'],capture_output=True,text=True,timeout=20)
    assert help_result.returncode==0 and '--case' in help_result.stdout
    invalid=subprocess.run([sys.executable,str(ENTRY),'--timeout','0','--output',str(tmp_path/'x.json')],
                           capture_output=True,text=True,timeout=20)
    assert invalid.returncode!=0


def test_configuration_boundaries(monkeypatch):
    from minivllm.config import DeviceConfig, ModelConfig, ParallelConfig, VllmConfig
    from minivllm.executor import MultiprocExecutor, get_executor_class
    model=ModelConfig('unused',hf_config={'num_attention_heads':4,'num_key_value_heads':2,
                                          'num_hidden_layers':2},enforce_eager=True)
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1')
    cfg=VllmConfig(model,device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2))
    assert get_executor_class(cfg) is MultiprocExecutor
    with pytest.raises(NotImplementedError, match='DP'):
        VllmConfig(model,device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(data_parallel_size=2))
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','0')
    with pytest.raises(NotImplementedError, match='V2'):
        VllmConfig(model,device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2))


def test_cuda_tensor_config_rejected_before_spawn():
    script = '''
import os
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
import torch
from minivllm.config import DeviceConfig,ModelConfig,ParallelConfig,VllmConfig
from minivllm.executor import MultiprocExecutor
m=ModelConfig('unused',hf_config={'num_attention_heads':4,'num_key_value_heads':2,'num_hidden_layers':2,'illegal':torch.zeros(1,device='cuda')},enforce_eager=True)
c=VllmConfig(m,device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2))
try: MultiprocExecutor(c)
except ValueError as e: assert 'CUDA 张量' in str(e)
else: raise AssertionError('CUDA tensor must be rejected')
'''
    p=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr


def test_distributed_rejects_out_of_range_local_rank_before_group_init():
    from minivllm.distributed import init_distributed_environment
    with pytest.raises(ValueError, match='local_rank.*world_size'):
        init_distributed_environment(world_size=2,rank=0,local_rank=2,device_index=0)
