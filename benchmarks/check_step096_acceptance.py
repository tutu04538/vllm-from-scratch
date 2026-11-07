#!/usr/bin/env python3
"""096 TP linear layer frozen-vLLM differential on 1/2/4 GPUs."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'tests/step096/step096_rank_worker.py'


def execute(world, out, timeout):
    path=out/f'tp{world}';path.mkdir(parents=True,exist_ok=True)
    for f in path.glob('rank*.json'):f.unlink()
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',str(WORKER),'--out',str(path)]
    start=time.monotonic()
    with (path/'torchrun.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    ranks=[json.loads((path/f'rank{r}.json').read_text()) if (path/f'rank{r}.json').exists() else {'rank':r,'complete':False} for r in range(world)]
    return {'world':world,'command':cmd,'exit_code':code,'passed':code==0 and all(r['complete'] for r in ranks),'ranks':ranks,'elapsed_seconds':time.monotonic()-start,'log':str(path/'torchrun.log')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['all','1','2','4'],default='all');p.add_argument('--timeout',type=int,default=180);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.timeout<=0:p.error('--timeout must be positive')
    worlds=(1,2,4) if a.case=='all' else (int(a.case),)
    cases=[execute(w,a.output.parent/(a.output.stem+'_ranks'),a.timeout) for w in worlds]
    result={'status':'passed' if all(c['passed'] for c in cases) else 'failed','cases':cases}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(cases),'output':str(a.output)}))
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())
