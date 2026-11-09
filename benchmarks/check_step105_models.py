"""Validate 105 real Graph model cases against the registered 103 frozen reference."""
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
    base=a.output.parent/'graph_models';base.mkdir(parents=True,exist_ok=True);cases=[]
    for model,tp,batch in CASES:
        name=f'{model}_tp{tp}_b{batch}';directory=base/name;directory.mkdir(exist_ok=True)
        probe=[sys.executable,'benchmarks/step105_model_probe.py','--model',model,'--tp',str(tp),'--batch',str(batch),'--output',str(directory)]
        stages=[run(probe,directory/'run.log',a.timeout)]
        if stages[0]['exit_code']==0:
            compare=[sys.executable,'benchmarks/step103_compare.py','--model',model,'--tp',str(tp),'--batch',str(batch),'--mini-dir',str(directory),'--output',str(directory/'compare.json')]
            stages.append(run(compare,directory/'compare.log',600))
        valid=all(x['exit_code']==0 for x in stages)
        graph=[]
        if valid:
            summary=json.loads((directory/'summary.json').read_text())
            graph=summary['graph']
            import torch
            rank_rows=[]
            for rank in range(tp):
                rows=torch.load(directory/f'run0_rank{rank}.pt',map_location='cpu',weights_only=True)
                rank_rows.append({(x['request_id'],x['position']):x['sampled'] for x in rows})
            progress=[x['sampling_progress'] for x in summary['runs'][0]['ranks']]
            valid=(len(graph)==tp and all(x['mode']=='FULL_DECODE_ONLY' and x['captured']>0 and x['replays']>0 for x in graph)
                   and all(x['selections']==graph[0]['selections'] and x['buckets']==graph[0]['buckets'] for x in graph)
                   and all(x==rank_rows[0] for x in rank_rows) and all(x==progress[0] for x in progress))
        result={'case':name,'status':'passed' if valid else 'failed','graph':graph,'stages':stages}
        cases.append(result);print(json.dumps(result),flush=True)
        a.output.write_text(json.dumps({'status':'running','cases':cases},indent=2)+'\n')
    result={'status':'passed' if all(x['status']=='passed' for x in cases) else 'failed','cases':cases}
    a.output.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
