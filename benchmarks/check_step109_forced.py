"""Replay frozen real-model history in eager and Graph, compare every rank."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=int,default=1800);a=p.parse_args()
 if a.timeout<=0:p.error('timeout must be positive')
 matrix=json.loads(a.matrix.read_text())
 if matrix['status']!='passed':raise ValueError('natural reference matrix has not passed')
 source=a.matrix.parent/(a.matrix.stem+'_runs');out=a.output.resolve();folder=out.parent/(out.stem+'_runs');folder.mkdir(parents=True,exist_ok=True)
 executions=[]
 def run(name,cmd,frozen=False):
  env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
  if frozen:env['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
  log=folder/f'{name}.log';start=time.monotonic()
  with log.open('w') as f:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=proc.wait(timeout=a.timeout)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    code=124
  record={'name':name,'command':cmd,'frozen_core_in_process':frozen,'exit_code':code,'seconds':time.monotonic()-start,'log':str(log)}
  executions.append(record);print(json.dumps(record),flush=True)
  out.write_text(json.dumps({'status':'running','executions':executions},indent=2)+'\n')
  if code:raise RuntimeError(f'{name} failed: {log}')
 try:
  for model in ('8b','14b'):
   for mode in ('eager','graph'):
    name=f'{model}_tp2_b1_k2_{mode}';prior=source/f'{name}_ref'
    ref,repeat,mini=(folder/f'{name}_{suffix}' for suffix in ('ref','repeat','mini'));contract=folder/f'{name}_contract.json'
    common=['--model',model,'--tp','2','--batch','1','--k','2','--mode',mode,'--forced-from',str(prior)]
    for label,directory in (('ref',ref),('repeat',repeat)):
     run(name+'_'+label,[sys.executable,str(ROOT/'benchmarks/step109_forced_probe.py'),'--implementation','frozen',*common,'--output',str(directory)],True)
    run(name+'_register',[sys.executable,str(ROOT/'benchmarks/step109_compare.py'),'--reference',str(ref),'--other',str(repeat),'--contract',str(contract),'--output',str(folder/f'{name}_repeat.json'),'--register'])
    run(name+'_mini',[sys.executable,str(ROOT/'benchmarks/step109_forced_probe.py'),'--implementation','mini',*common,'--output',str(mini)])
    run(name+'_compare',[sys.executable,str(ROOT/'benchmarks/step109_compare.py'),'--reference',str(ref),'--other',str(mini),'--contract',str(contract),'--output',str(folder/f'{name}_comparison.json')])
  status='passed'
 except RuntimeError as exc:print(str(exc),file=sys.stderr);status='failed'
 out.write_text(json.dumps({'status':status,'executions':executions},indent=2)+'\n');return int(status!='passed')
if __name__=='__main__':raise SystemExit(main())
