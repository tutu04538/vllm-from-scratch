#!/usr/bin/env python3
"""101 independent production-worker tiny and real Qwen3 TP checks."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASES=('tiny','8b1','8b2','8b4','14b2')


def stage(cmd,log,timeout):
 start=time.monotonic()
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=124
 return {'command':cmd,'exit_code':code,'seconds':round(time.monotonic()-start,3),'log':str(log)}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['all',*CASES],default='all');p.add_argument('--timeout',type=int,default=1800);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.timeout<=0:p.error('--timeout must be positive')
 a.output.parent.mkdir(parents=True,exist_ok=True);results=[]
 for case in CASES if a.case=='all' else [a.case]:
  out=a.output.parent/(a.output.stem+'_'+case);out.mkdir(parents=True,exist_ok=True);stages=[]
  if case=='tiny':
   stages.append(stage([sys.executable,'-m','pytest','tests/step101','-q','-s','-ra'],out/'tiny.log',min(a.timeout,600)))
  else:
   model='14b' if case.startswith('14b') else '8b';tp=int(case[-1]);probe=ROOT/'tests/step101/step101_probe.py'
   cmd=[sys.executable,str(probe),'--model',model,'--tp',str(tp),'--output',str(out)]
   if model=='14b':cmd+=['--load-only']
   stages.append(stage(cmd,out/'load.log',a.timeout))
   if stages[-1]['exit_code']==0:
    cmd=[sys.executable,'benchmarks/check_step101_manifest.py','--model',model,'--tp',str(tp),'--probe-dir',str(out),'--output',str(out/'manifest_audit.json')]
    stages.append(stage(cmd,out/'manifest.log',a.timeout))
   if model=='8b' and all(s['exit_code']==0 for s in stages):
    cmd=[sys.executable,'benchmarks/check_step101_numerics.py','--tp',str(tp),'--probe-dir',str(out),'--output',str(out/'compare.json')]
    stages.append(stage(cmd,out/'numerics.log',a.timeout))
  results.append({'case':case,'passed':all(s['exit_code']==0 for s in stages),'stages':stages})
  print(json.dumps({'case':case,'passed':results[-1]['passed']}),flush=True)
 result={'status':'passed' if all(r['passed'] for r in results) else 'failed','cases':results};a.output.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
