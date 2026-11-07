#!/usr/bin/env python3
"""097 QKV/Merged TP slicing: tiny frozen-vLLM differential and real Qwen layer-0 weights."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'tests/step097/step097_rank_worker.py'
CASES={'tiny1':('tiny',1),'tiny2':('tiny',2),'tiny4':('tiny',4),
       '8b2':('8b',2),'8b4':('8b',4),'14b2':('14b',2),'14b4':('14b',4)}

def execute(case,out,timeout):
    which,world=CASES[case];path=out/case;path.mkdir(parents=True,exist_ok=True)
    for f in path.glob('rank*.json'):f.unlink()
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',str(WORKER),'--case',which,'--out',str(path)]
    started=time.monotonic()
    with (path/'torchrun.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    ranks=[json.loads((path/f'rank{i}.json').read_text()) if (path/f'rank{i}.json').exists() else {'rank':i,'complete':False} for i in range(world)]
    return {'case':case,'world':world,'command':cmd,'exit_code':code,'passed':code==0 and all(r['complete'] for r in ranks),'ranks':ranks,'elapsed_seconds':time.monotonic()-started,'log':str(path/'torchrun.log')}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['all',*CASES],default='all');p.add_argument('--timeout',type=int,default=180);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.timeout<=0:p.error('--timeout must be positive')
    cases=list(CASES) if a.case=='all' else [a.case]
    values=[execute(c,a.output.parent/(a.output.stem+'_ranks'),a.timeout) for c in cases]
    result={'status':'passed' if all(v['passed'] for v in values) else 'failed','cases':values}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(values),'output':str(a.output)}));return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
