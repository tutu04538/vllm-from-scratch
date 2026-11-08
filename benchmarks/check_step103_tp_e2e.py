"""103 serial real-model matrix against pre-registered frozen reference."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmarks/results/step103'
CASES={f'{model}_tp{tp}_b{b}':(model,tp,b) for model,tps in [('8b',[1,2,4]),('14b',[2,4])] for tp in tps for b in [1,4]}


def stage(cmd,log,timeout):
 start=time.monotonic()
 with log.open('w') as stream:
  proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   rc=124
 return {'command':cmd,'exit_code':rc,'seconds':round(time.monotonic()-start,3),'log':str(log)}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['all',*CASES],default='all')
 p.add_argument('--timeout',type=int,default=1800);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.timeout<=0:p.error('timeout must be positive')
 contract=OUT/'numerics_contract.json'
 if not contract.exists():p.error('先独立采集全部冻结参考并登记 numerics_contract.json')
 registered=json.loads(contract.read_text())
 if registered['status']!='registered':p.error('冻结参考历史不一致，不能运行 mini 验收')
 a.output.parent.mkdir(parents=True,exist_ok=True);results=[]
 for name in CASES if a.case=='all' else [a.case]:
  model,tp,b=CASES[name];dest=OUT/f'mini_{name}';dest.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'benchmarks/step103_mini_probe.py','--model',model,'--tp',str(tp),'--batch',str(b),'--output',str(dest)]
  stages=[stage(cmd,dest/'run.log',a.timeout)]
  if not stages[0]['exit_code']:
   cmd=[sys.executable,'benchmarks/step103_compare.py','--model',model,'--tp',str(tp),'--batch',str(b),'--mini-dir',str(dest),'--output',str(dest/'compare.json')]
   stages.append(stage(cmd,dest/'compare.log',min(a.timeout,600)))
  item={'case':name,'status':'passed' if all(s['exit_code']==0 for s in stages) else 'failed','stages':stages}
  results.append(item);print(json.dumps({'case':name,'status':item['status']}),flush=True)
  a.output.write_text(json.dumps({'status':'running','cases':results},indent=2)+'\n')
 result={'status':'passed' if all(r['status']=='passed' for r in results) else 'failed','cases':results}
 a.output.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
