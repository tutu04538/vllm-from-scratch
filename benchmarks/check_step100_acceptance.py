#!/usr/bin/env python3
"""Run 100 real-TP sampling, state mirrors and frozen Gumbel checks."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TEST='tests/step100/test_step100_sampling.py'
CASES={f'{mode}{tp}':f'{TEST}::test_real_tp_sampling_and_mirrors[{mode}-{tp}]' for mode in ('greedy','random','preempt') for tp in (2,4)}
CASES.update({'gumbel':f'{TEST}::test_frozen_gumbel_and_distribution','seed':f'{TEST}::test_model_seed_reset_and_logit_bias_rejection'})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['all',*CASES],default='all');p.add_argument('--timeout',type=int,default=240);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.timeout<=0:p.error('--timeout must be positive')
    a.output.parent.mkdir(parents=True,exist_ok=True);results=[]
    for case in CASES if a.case=='all' else [a.case]:
        cmd=[sys.executable,'-m','pytest',CASES[case],'-q','-s','-ra'];log=a.output.parent/f'{a.output.stem}_{case}.log';start=time.monotonic()
        with log.open('w') as f:
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                 'STEP100_TRACE_DIR':str(a.output.parent/(a.output.stem+'_traces'))}
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        results.append({'case':case,'command':cmd,'exit_code':code,'seconds':round(time.monotonic()-start,3),'log':str(log)})
    passed=all(r['exit_code']==0 for r in results)
    a.output.write_text(json.dumps({'status':'passed' if passed else 'failed','cases':results},indent=2)+'\n')
    print(json.dumps({'status':'passed' if passed else 'failed','output':str(a.output)}));return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
