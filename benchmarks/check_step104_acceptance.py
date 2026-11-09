#!/usr/bin/env python3
"""Run graph-captured NCCL collectives serially on real local GPUs."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'tests/step104/step104_rank_worker.py'


def execute(world,failure,out,timeout):
    directory=out/f'tp{world}{"_failure" if failure else ""}'
    directory.mkdir(parents=True,exist_ok=True)
    for pattern in ('rank*.json','cycle*.done','store*'):
        for old in directory.glob(pattern):old.unlink()
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',str(WORKER),'--out',str(directory.resolve())]
    if failure:cmd.append('--failure')
    start=time.monotonic();log=directory/'torchrun.log'
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    ranks=[json.loads(p.read_text()) for p in sorted(directory.glob('rank*.json'))]
    passed=(code==0 and len(ranks)==world and all(r['complete'] for r in ranks)) if not failure else (code not in (0,124) and 'step104 injected rank failure' in log.read_text())
    return {'world':world,'failure':failure,'passed':passed,'exit_code':code,'seconds':round(time.monotonic()-start,3),'command':cmd,'ranks':ranks,'log':str(log)}


def execute_executor(out, timeout, lifecycle=False):
    out.mkdir(parents=True,exist_ok=True)
    name='engine_lifecycle' if lifecycle else 'executor_failure'
    log=out/f'{name}.log'
    cmd=[sys.executable,str(ROOT/f'tests/step104/step104_{name}.py')]
    start=time.monotonic()
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    return {'case':name,'passed':code==0,'exit_code':code,'seconds':round(time.monotonic()-start,3),'command':cmd,'log':str(log)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['all','tp2','tp4','failure','executor_failure','engine_lifecycle'],default='all');p.add_argument('--timeout',type=int,default=120);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.timeout<=0:p.error('--timeout must be positive')
    cases={'tp2':(2,False),'tp4':(4,False),'failure':(2,True)}
    chosen=cases.values() if a.case=='all' else ([] if a.case in ('executor_failure','engine_lifecycle') else [cases[a.case]])
    results=[execute(world,fail,a.output.parent/(a.output.stem+'_ranks'),a.timeout) for world,fail in chosen]
    if a.case in ('all','executor_failure'):
        results.append(execute_executor(a.output.parent/(a.output.stem+'_ranks'),a.timeout))
    if a.case in ('all','engine_lifecycle'):
        results.append(execute_executor(a.output.parent/(a.output.stem+'_ranks'),a.timeout,lifecycle=True))
    result={'status':'passed' if all(x['passed'] for x in results) else 'failed','cases':results}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'cases':len(results)}));return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
