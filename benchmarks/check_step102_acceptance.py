"""102 serial isolated GPU acceptance: lifecycle, cancellation and failure."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = {'lifecycle':'test_prefix_and_preemption_tp2_tp4',
         'cancel':'test_cancel_and_slot_reuse', 'failure':'test_failed_rank_never_commits',
         'capacity':'test_heterogeneous_rank_capacity_and_rejection',
         'frozen':'test_frozen_scheduler_preemption_protocol',
         'prefix_compare':'test_prefix_hit_matches_cold_output'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['all',*CASES],default='all')
    p.add_argument('--timeout',type=int,default=900,help='Maximum seconds per isolated pytest case')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.timeout<=0: p.error('timeout must be positive')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cases=CASES if args.case=='all' else {args.case:CASES[args.case]}
    results=[]
    for case,test in cases.items():
        log=args.output.parent/f'{args.output.stem}_{case}.log'
        command=[sys.executable,'-m','pytest',f'tests/step102/test_step102_lifecycle.py::{test}','-q','-ra']
        start=time.monotonic()
        with log.open('w') as out:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                     'STEP102_TRACE_DIR':str((args.output.parent / (args.output.stem+'_traces')).resolve())})
            try: code=proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        results.append({'case':case,'command':command,'exit_code':code,'seconds':round(time.monotonic()-start,3),'log':str(log)})
        print(json.dumps({'case':case,'exit_code':code}),flush=True)
        if code: break
    result={'status':'passed' if len(results)==len(cases) and all(r['exit_code']==0 for r in results) else 'failed','cases':results}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':
    sys.exit(main())
