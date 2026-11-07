#!/usr/bin/env python3
"""Run 099 frozen-source differential on real TP1/2/4 and local negatives."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASES={'tp1':'tests/step099/test_step099_vocab.py::test_frozen_and_real_ranks[1]',
       'tp2':'tests/step099/test_step099_vocab.py::test_frozen_and_real_ranks[2]',
       'tp4':'tests/step099/test_step099_vocab.py::test_frozen_and_real_ranks[4]',
       'negative':'tests/step099/test_step099_vocab.py::test_tp1_padding_weight_validation_and_no_pad_sampling'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['all',*CASES],default='all');p.add_argument('--timeout',type=int,default=240);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.timeout<=0:p.error('--timeout must be positive')
    a.output.parent.mkdir(parents=True,exist_ok=True);results=[]
    for case in CASES if a.case=='all' else [a.case]:
        cmd=[sys.executable,'-m','pytest',CASES[case],'-q','-s','-ra'];log=a.output.parent/f'{a.output.stem}_{case}.log';start=time.monotonic()
        with log.open('w') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        golden=None
        if case.startswith('tp') and code==0:
            for line in log.read_text().splitlines():
                if line.startswith('[{'):
                    records=json.loads(line)
                    golden=a.output.parent/f'{a.output.stem}_{case}_golden.json'
                    golden.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
                    break
            if golden is None:code=1
        results.append({'case':case,'command':cmd,'exit_code':code,'elapsed_seconds':round(time.monotonic()-start,3),'log':str(log),'golden_fixture':str(golden) if golden else None})
    passed=all(x['exit_code']==0 for x in results)
    a.output.write_text(json.dumps({'status':'passed' if passed else 'failed','cases':results},indent=2)+'\n')
    print(json.dumps({'status':'passed' if passed else 'failed','output':str(a.output)}));return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
