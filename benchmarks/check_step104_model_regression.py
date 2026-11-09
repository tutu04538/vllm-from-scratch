"""Re-run 103 real model cases against its pre-registered frozen reference."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASES=[(m,tp,b) for m,tps in [('8b',[1,2,4]),('14b',[2,4])] for tp in tps for b in (1,4)]


def run(cmd,log,timeout):
    start=time.monotonic()
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    return {'command':cmd,'exit_code':code,'seconds':round(time.monotonic()-start,3),'log':str(log)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timeout',type=int,default=1800);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.timeout<=0:p.error('--timeout must be positive')
    base=a.output.parent/'real_model_regression';base.mkdir(parents=True,exist_ok=True);cases=[]
    for model,tp,batch in CASES:
        name=f'{model}_tp{tp}_b{batch}';directory=base/name;directory.mkdir(exist_ok=True)
        probe=[sys.executable,'benchmarks/step103_mini_probe.py','--model',model,'--tp',str(tp),'--batch',str(batch),'--output',str(directory)]
        stages=[run(probe,directory/'run.log',a.timeout)]
        if stages[0]['exit_code']==0:
            compare=[sys.executable,'benchmarks/step103_compare.py','--model',model,'--tp',str(tp),'--batch',str(batch),'--mini-dir',str(directory),'--output',str(directory/'compare.json')]
            stages.append(run(compare,directory/'compare.log',600))
        result={'case':name,'status':'passed' if all(x['exit_code']==0 for x in stages) else 'failed','stages':stages}
        cases.append(result);print(json.dumps(result),flush=True)
        a.output.write_text(json.dumps({'status':'running','cases':cases},indent=2)+'\n')
    result={'status':'passed' if all(x['status']=='passed' for x in cases) else 'failed','cases':cases}
    a.output.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
