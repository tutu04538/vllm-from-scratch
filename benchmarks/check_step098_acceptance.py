#!/usr/bin/env python3
"""Run 098 source differential and real-GPU TP checks with bounded subprocesses."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CASES={'source':'tests/step098/test_step098_capacity.py',
       'tp2':'tests/step098/test_step098_gpu.py::test_real_rank_cache[2-8,6]',
       'tp4':'tests/step098/test_step098_gpu.py::test_real_rank_cache[4-8,7,6,8]',
       'failure':'tests/step098/test_step098_gpu.py::test_rank_budget_failure'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['all',*CASES],default='all')
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.timeout<=0:p.error('--timeout must be positive')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    for case in CASES if a.case=='all' else [a.case]:
        cmd=[sys.executable,'-m','pytest',CASES[case],'-q','-s','-ra']
        log=a.output.parent/f'{a.output.stem}_{case}.log'
        start=time.monotonic()
        with log.open('w') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        results.append({'case':case,'command':cmd,'exit_code':code,'seconds':round(time.monotonic()-start,3),'log':str(log)})
    passed=all(x['exit_code']==0 for x in results)
    a.output.write_text(json.dumps({'status':'passed' if passed else 'failed','cases':results},indent=2)+'\n')
    print(json.dumps({'status':'passed' if passed else 'failed','output':str(a.output)}))
    return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
