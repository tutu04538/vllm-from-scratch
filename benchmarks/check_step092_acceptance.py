#!/usr/bin/env python3
"""092 单机 NCCL 组与原语验收；torchrun 仅作为测试启动器。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'tests/step092/step092_rank_worker.py'


def execute(case, out, timeout):
    world,tp,pp,fail,oracle=case
    suffix = '_oracle' if oracle else ('_failure' if fail is not None else '')
    path=out / f'w{world}_tp{tp}_pp{pp}{suffix}'
    path.mkdir(parents=True,exist_ok=True)
    for old in path.glob('rank*.json'): old.unlink()
    for old in path.glob('cycle*-rank*.done'): old.unlink()
    for old in path.glob('rendezvous*.store'): old.unlink()
    for old in path.glob('failure.store'): old.unlink()
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',
         str(WORKER),'--tp',str(tp),'--pp',str(pp),'--out',str(path), '--timeout',str(min(30,timeout))]
    if fail is not None: cmd += ['--failure-rank',str(fail)]
    if oracle: cmd += ['--oracle']
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(ROOT)}
    start=time.monotonic()
    with (path/'torchrun.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                              text=True,start_new_session=True)
        try:
            code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); proc.wait()
            code=124
    ranks=[]
    for rank in range(world):
        file=path/f'rank{rank}.json'
        ranks.append(json.loads(file.read_text()) if file.exists() else {'rank':rank,'complete':False,'error':'rank 未返回，可能被启动器终止'})
    ok=(code == 0 and all(r['complete'] for r in ranks)) if fail is None else (code not in (0,124) and not ranks[fail]['complete'])
    return {'case':{'world_size':world,'tp':tp,'pp':pp,'failure_rank':fail,'oracle':oracle},'command':cmd,'exit_code':code,
            'passed':ok,'elapsed_seconds':time.monotonic()-start,'ranks':ranks,'log':str(path/'torchrun.log')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['all','1','2','4','mixed','failure','oracle'],default='all')
    p.add_argument('--timeout',type=int,default=120); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.timeout <= 0: p.error('--timeout must be positive')
    cases={'1':(1,1,1,None,False),'2':(2,2,1,None,False),'4':(4,4,1,None,False),'mixed':(4,2,2,None,False),'failure':(2,2,1,1,False),'oracle':(2,2,1,None,True)}
    chosen=list(cases.values()) if a.case=='all' else [cases[a.case]]
    base=a.output.parent/(a.output.stem+'_ranks'); start=time.monotonic()
    records=[execute(case,base,a.timeout) for case in chosen]
    manifest=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/源码清单.json'
    import torch
    result={'status':'passed' if all(r['passed'] for r in records) else 'failed',
            'elapsed_seconds':time.monotonic()-start,'cases':records,
            'environment':{'python':sys.version.split()[0], 'torch':torch.__version__,
                           'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')},
            'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(records),'output':str(a.output)},ensure_ascii=False))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__': sys.exit(main())
