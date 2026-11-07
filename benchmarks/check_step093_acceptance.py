#!/usr/bin/env python3
"""093 多进程 Worker 启动与设备所有权验收，不执行多卡模型。"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all','2','4','device-ids','fail-device','fail-load','production'], default='all')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.timeout <= 0: parser.error('--timeout must be positive')
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    import torch
    from minivllm.config import DeviceConfig, ModelConfig, ParallelConfig, VllmConfig
    from minivllm.executor import MultiprocExecutor, get_executor_class
    from minivllm.testing.step093_probe import ProbeWorker, FailInitWorker, FailLoadWorker
    def expired(signum, frame): raise TimeoutError(f'验收超过 {args.timeout} 秒')
    signal.signal(signal.SIGALRM, expired)
    cases = ['2','4','device-ids','fail-device','fail-load','production'] if args.case == 'all' else [args.case]
    result={'status':'failed','cases':[],'parent_pid':os.getpid(),'start_method':'spawn',
            'parent_cuda_initialized_before':torch.cuda.is_initialized(),
            'environment':{'python':sys.version.split()[0], 'torch':torch.__version__,
                           'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')},
            'source_manifest_sha256':hashlib.sha256((ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/源码清单.json').read_bytes()).hexdigest()}
    start=time.monotonic()
    try:
        for case in cases:
            signal.alarm(args.timeout)
            n=4 if case=='4' else 2
            model=ModelConfig('unused',hf_config={'num_attention_heads':32,'num_key_value_heads':8,
                              'num_hidden_layers':36},enforce_eager=True)
            ids=(1,0) if case == 'device-ids' else None
            cfg=VllmConfig(model,device_config=DeviceConfig('cuda'),
                           parallel_config=ParallelConfig(tensor_parallel_size=n,device_ids=ids))
            assert get_executor_class(cfg) is MultiprocExecutor
            selected={'fail-device':FailInitWorker,'fail-load':FailLoadWorker,'production':None}.get(case,ProbeWorker)
            item={'case':case,'workers':n,'status':'failed','attempts':[]}
            for _ in range(2 if case in ('2','4') else 1):
                children_before={p.pid for p in multiprocessing.active_children()}
                dirs_before=set(Path(tempfile.gettempdir()).glob('minivllm-mp-*'))
                executor=None
                try:
                    executor=MultiprocExecutor(cfg,_worker_cls=selected,_startup_timeout=args.timeout)
                    if case in ('fail-device','fail-load','production'):
                        raise AssertionError('预期初始化失败')
                    info=list(executor.ready_workers)
                    assert len({w['pid'] for w in info}) == n
                    assert len({w['device_uuid'] for w in info}) == n
                    assert all(not w['model_loaded'] for w in info)
                    assert all(w['group_local_rank']==w['local_rank'] and w['group_device_index']==w['device_index'] for w in info)
                    if ids is not None:
                        assert tuple(w['device_index'] for w in info)==ids
                    pids=[h.process.pid for h in executor.workers]
                    rendezvous=executor._rendezvous.name
                    executor.shutdown()
                    assert not Path(rendezvous).exists()
                    assert all(not h.process.is_alive() for h in executor.workers)
                    item['attempts'].append({'ready':info,'pids':pids,'rendezvous_cleaned':True})
                except (RuntimeError, NotImplementedError) as exc:
                    expected={'fail-device':('rank 1','init_device'), 'fail-load':('rank 1','load_model'),
                              'production':('load_model','参数分片')}.get(case)
                    if expected is None or not all(s in str(exc) for s in expected): raise
                    item['attempts'].append({'expected_error':str(exc)})
                finally:
                    if executor is not None: executor.shutdown()
                assert {p.pid for p in multiprocessing.active_children()} <= children_before
                assert set(Path(tempfile.gettempdir()).glob('minivllm-mp-*')) <= dirs_before
                item['attempts'][-1]['children_reaped']=True
                item['attempts'][-1]['rendezvous_cleaned']=True
            item['status']='passed'
            result['cases'].append(item)
            signal.alarm(0)
        result['status']='passed'
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        signal.alarm(0)
        result['elapsed_seconds']=time.monotonic()-start
        result['parent_cuda_initialized_after']=torch.cuda.is_initialized()
        if result['parent_cuda_initialized_before'] or result['parent_cuda_initialized_after']:
            result['status']='failed'
            result['error']='父进程不应初始化 CUDA'
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'error':result.get('error'), 'output':str(args.output)},ensure_ascii=False))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__': sys.exit(main())
