"""Collect per-rank workspace for all 103 cases and constrained random sampling."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmarks/results/step103'
CASES=[(model,tp,b) for model,tps in [('8b',[1,2,4]),('14b',[2,4])] for tp in tps for b in [1,4]]


def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['memory','random','forced','all'],default='all');p.add_argument('--timeout',type=int,default=1800);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.timeout<=0:p.error('timeout must be positive')
 tasks=[]
 if a.mode == 'forced':
  tasks.extend((engine,model,tp,b,'forced') for model,tp,b in CASES for engine in ('frozen','mini'))
 if a.mode in ('memory','all'):
  tasks.extend((engine,model,tp,b,False) for model,tp,b in CASES for engine in ('frozen','mini'))
 if a.mode in ('random','all'):
  tasks.extend((engine,model,tp,4,True) for model,tp in [('8b',2),('14b',4)] for engine in ('frozen','mini'))
 a.output.parent.mkdir(parents=True,exist_ok=True);items=[]
 for engine,model,tp,b,random in tasks:
  forced = random == 'forced'
  is_random = random is True
  name=('forced_' if forced else 'aux_')+f'{engine}_{model}_tp{tp}_b{b}'+('_random' if is_random else '')
  directory=OUT/name;log=OUT/(name+'.log')
  cmd=[sys.executable,'benchmarks/step103_aux_probe.py','--engine',engine,'--model',model,'--tp',str(tp),'--batch',str(b),'--output',str(directory)]
  if is_random:cmd.append('--random')
  if forced:cmd.append('--forced')
  start=time.monotonic()
  with log.open('w') as stream:
   proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=proc.wait(timeout=a.timeout)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    rc=124
  item={'engine':engine,'model':model,'tp':tp,'batch':b,'random':is_random,'forced':forced,'exit_code':rc,'seconds':round(time.monotonic()-start,3),'summary':str(directory/'summary.json'),'log':str(log)}
  items.append(item);print(json.dumps(item),flush=True)
  a.output.write_text(json.dumps({'status':'running','cases':items},indent=2)+'\n')
  if rc:break
 result={'status':'passed' if len(items)==len(tasks) and all(x['exit_code']==0 for x in items) else 'failed','cases':items}
 a.output.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
