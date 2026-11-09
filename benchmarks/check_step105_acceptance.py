#!/usr/bin/env python3
"""Independent tiny-model TP Graph/eager, padding and lifecycle acceptance."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run(cmd,log,timeout):
 start=time.monotonic()
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   rc=124
 return {'command':cmd,'exit_code':rc,'seconds':round(time.monotonic()-start,3),'log':str(log)}


def compare(tp,directory,dynamic):
 import torch
 eager=json.loads((directory/'eager/summary.json').read_text());graph=json.loads((directory/'graph/summary.json').read_text())
 assert eager['outputs']==graph['outputs'] and eager['steps']==graph['steps']
 traces=[];comparisons=0;max_abs=0.
 for rank in range(tp):
  er=torch.load(directory/f'eager/trace_rank{rank}.pt',map_location='cpu',weights_only=True)
  gr=torch.load(directory/f'graph/trace_rank{rank}.pt',map_location='cpu',weights_only=True)
  # A final scheduler-only cleanup tick may not call Runner.sample.
  assert len(er)==len(gr) and eager['steps']-len(er) in (0,1);traces.append(gr)
  for e,g in zip(er,gr):
   assert [r['request_id'] for r in e['rows']]==[r['request_id'] for r in g['rows']]
   assert all(slot==-1 for slot in g['slots'][g['num_tokens']:].tolist())
   for x,y in zip(e['rows'],g['rows']):
    assert x['position']==y['position'] and x['sampled']==y['sampled']
    for a,b in [(x['hidden'],y['hidden']),(x['logits'],y['logits']),*zip(x['kv'],y['kv'])]:
     torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5);max_abs=max(max_abs,float((a-b).abs().max()));comparisons+=1
    assert y['physical_blocks'][y['position']//16]*16+y['position']%16 in g['slots'].tolist()
 stats=graph['ranks'];assert all(before['addresses']==after['addresses'] and before['graph_count']==after['graph_count'] for before,after in zip(graph['before'],stats))
 assert all(r['mode']=='FULL_DECODE_ONLY' and r['buckets']==[1,2,4] and r['replays']>0 for r in stats)
 assert all(r['selections']==stats[0]['selections'] for r in stats)
 # TP ranks consume the same request/position/slot protocol and sample the same token.
 for records in zip(*traces):
  compact=lambda r: [(x['request_id'],x['position'],x['sampled'],x['resident_slot'],x['physical_blocks']) for x in r['rows']]
  assert all(compact(row)==compact(records[0]) and torch.equal(row['slots'],records[0]['slots']) for row in records)
 full=[s for s in stats[0]['selections'] if s['mode']=='FULL']
 if dynamic:
  counts=[s['num_reqs'] for s in full]
  changes=[value for i,value in enumerate(counts) if i==0 or value!=counts[i-1]]
  assert any(changes[i:i+3]==[3,4,3] for i in range(len(changes)-2)),counts
  records=traces[0];seen={}
  for rec in records:
   for row in rec['rows']:seen.setdefault(row['request_id'],[]).append(row)
  assert seen['1'][0]['resident_slot']==seen['4'][0]['resident_slot'] or seen['0'][0]['resident_slot']==seen['4'][0]['resident_slot']
  assert any(len(row['physical_blocks'])>=2 for rows in seen.values() for row in rows)
 return {'status':'passed','tp':tp,'dynamic':dynamic,'steps':eager['steps'],'comparisons':comparisons,'max_abs':max_abs,'ranks':stats}


def execute(tp,dynamic,directory,timeout):
 directory.mkdir(parents=True,exist_ok=True);stages=[]
 for mode in ('eager','graph'):
  cmd=[sys.executable,'tests/step105/step105_tiny_probe.py','--mode',mode,'--tp',str(tp),'--batch','3','--output',str(directory/mode)]
  if dynamic:cmd.append('--dynamic')
  if mode=='graph':cmd.append('--poison-padding')
  stage=run(cmd,directory/f'{mode}.log',timeout);stages.append(stage)
  if stage['exit_code']:return {'status':'failed','stages':stages}
 try:
  result=compare(tp,directory,dynamic);result['stages']=stages;return result
 except Exception as exc:
  import traceback
  return {'status':'failed','stages':stages,'error':str(exc),'traceback':traceback.format_exc()}


def execute_random(directory,timeout):
 import torch
 directory.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,'tests/step105/step105_tiny_probe.py','--mode','graph','--tp','2','--batch','3',
      '--random','--poison-padding','--output',str(directory/'output')]
 stage=run(cmd,directory/'run.log',timeout)
 if stage['exit_code']:return {'status':'failed','case':'random','stage':stage}
 try:
  summary=json.loads((directory/'output/summary.json').read_text())
  traces=[torch.load(directory/f'output/trace_rank{rank}.pt',map_location='cpu',weights_only=True) for rank in range(2)]
  assert len(traces[0])==len(traces[1]);checks=0
  for a,b in zip(*traces):
   assert len(a['rows'])==len(b['rows'])
   for x,y in zip(a['rows'],b['rows']):
    assert (x['request_id'],x['position'],x['sampled'])==(y['request_id'],y['position'],y['sampled'])
    for row in (x,y):
     # Independent CPU calculation of the sampler's temperature → min_p → top-k → top-p eligibility.
     import math
     logits=row['logits'].float()/0.8
     logits=logits.masked_fill(logits < logits.max()+math.log(0.05),float('-inf'))
     ordered,indices=logits.sort()
     ordered=ordered.masked_fill(ordered < ordered[-5],float('-inf'))
     mask=ordered.softmax(dim=0).cumsum(dim=0) <= 0.05
     mask[-1]=False
     allowed=indices[~mask & torch.isfinite(ordered)]
     assert row['sampled'] in allowed.tolist()
     checks+=1
  assert all(r['replays']>0 for r in summary['ranks']) and checks>0
  return {'status':'passed','case':'random','stage':stage,'rank_token_checks':checks,'ranks':summary['ranks']}
 except Exception as exc:
  import traceback
  return {'status':'failed','case':'random','stage':stage,'error':str(exc),'traceback':traceback.format_exc()}


def execute_lifecycle(name,directory,timeout):
 directory.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(ROOT/f'tests/step105/step105_{name}.py'),'--output',str(directory/'output')]
 stage=run(cmd,directory/'run.log',timeout)
 return {'status':'passed' if stage['exit_code']==0 else 'failed','stage':stage,'case':name}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['all','tp1','tp2','tp4','dynamic','engine_lifecycle','capture_failure','random'],default='all');p.add_argument('--timeout',type=int,default=180);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.timeout<=0:p.error('--timeout must be positive')
 all_cases={'tp1':(1,False),'tp2':(2,False),'tp4':(4,False),'dynamic':(2,True)}
 chosen=all_cases.items() if a.case=='all' else ([] if a.case in ('engine_lifecycle','capture_failure','random') else [(a.case,all_cases[a.case])])
 base=a.output.parent/(a.output.stem+'_ranks');results=[]
 for name,(tp,dynamic) in chosen:
  result=execute(tp,dynamic,base/name,a.timeout);result['case']=name;results.append(result);print(json.dumps({'case':name,'status':result['status']}),flush=True)
 if a.case in ('all','random'):
  result=execute_random(base/'random',a.timeout);results.append(result);print(json.dumps({'case':'random','status':result['status']}),flush=True)
 for name in ('engine_lifecycle','capture_failure'):
  if a.case in ('all',name):
   result=execute_lifecycle(name,base/name,a.timeout);results.append(result);print(json.dumps({'case':name,'status':result['status']}),flush=True)
 result={'status':'passed' if all(r['status']=='passed' for r in results) else 'failed','cases':results}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
