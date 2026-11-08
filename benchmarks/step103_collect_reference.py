"""Collect the full frozen reference matrix serially, before mini is measured."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/step103'
CASES=[(model,tp,b) for model,tps in [('8b',[1,2,4]),('14b',[2,4])] for tp in tps for b in [1,4]]

def main():
 results=[]
 for model,tp,b in CASES:
  name=f'reference_{model}_tp{tp}_b{b}'
  dest=OUT/name;log=OUT/f'{name}.log'
  cmd=[sys.executable,'benchmarks/step103_frozen_probe.py','--model',model,'--tp',str(tp),'--batch',str(b),'--repeats','2','--output',str(dest)]
  start=time.monotonic()
  with log.open('w') as f:
   proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=proc.wait(timeout=1800)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    rc=124
  item={'model':model,'tp':tp,'batch':b,'exit_code':rc,'seconds':round(time.monotonic()-start,3),'log':str(log)}
  results.append(item);print(json.dumps(item),flush=True)
  (OUT/'reference_matrix.json').write_text(json.dumps({'cases':results},indent=2)+'\n')
  if rc:return 1
 return 0
if __name__=='__main__':sys.exit(main())
