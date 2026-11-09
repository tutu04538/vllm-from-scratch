"""Run real Qwen3 Graph with fixed histories and compare each rank to 103 oracle."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.check_step105_models import CASES,run
from benchmarks.step103_compare import summary_metric
from benchmarks.step103_workload import prompts


def compare(model,tp,batch,directory):
 refdir=ROOT/f'benchmarks/results/step103/reference_{model}_tp{tp}_b{batch}'
 contract_path=ROOT/'benchmarks/results/step103/numerics_contract.json'
 registered=json.loads(contract_path.read_text());assert registered['status']=='registered'
 for file in [refdir/'summary.json',*[refdir/f'run0_rank{rank}.pt' for rank in range(tp)]]:
  key=str(file.relative_to(ROOT/'benchmarks/results/step103'))
  assert hashlib.sha256(file.read_bytes()).hexdigest()==registered['reference_sha256'][key],file
 refsummary=json.loads((refdir/'summary.json').read_text());refout=refsummary['runs'][0]['outputs']
 summary=json.loads((directory/'summary.json').read_text());out=summary['runs'][0]['outputs']
 histories=[item['tokens'][:4] for item in refout]
 assert [item['tokens'] for item in out]==histories
 contract=registered['thresholds']
 values=[];all_inputs=[];ids=prompts(model,batch)
 for rank in range(tp):
  reference=torch.load(refdir/f'run0_rank{rank}.pt',map_location='cpu',weights_only=True)
  got=torch.load(directory/f'run0_rank{rank}.pt',map_location='cpu',weights_only=True)
  def keyed(rows):
   ans={}
   for row in rows:
    req=int(row['request_id'].split('-',1)[0]);step=row['sequence_length']-len(ids[req]);ans[(req,step)]=row
   return ans
  expected,actual=keyed(reference),keyed(got);assert len(expected)==len(actual)==batch*4 and expected.keys()==actual.keys()
  for key in expected:
   x=actual[key];y=expected[key];req,step=key
   assert x['position']==y['position']==len(ids[req])-1+step
   for field,obj in [('hidden','hidden'),('logits','raw_logits'),('kv','kv')]:
    metrics=summary_metric(x[field],y[field]);limit=contract[f'{model}_b{batch}_step{step}_{obj}']
    values.append({'rank':rank,'request':req,'step':step,'object':obj,**metrics,
                   'passed':metrics['max_abs']<=limit['max_abs'] and metrics['relative_l2']<=limit['relative_l2']})
  trace=torch.load(summary['forced_ranks'][rank]['path'],map_location='cpu',weights_only=True);entries=[]
  for row in trace:
   req=int(row['request_id'].split('-',1)[0])
   for tok,pos,slot in zip(row['input_ids'],row['positions'],row['slot_mapping']):
    assert slot==row['block_table'][pos//16]*16+pos%16
    entries.append((req,pos,tok))
  all_inputs.append(sorted(entries))
 expected_input=sorted((req,pos,tok) for req in range(batch) for pos,tok in enumerate(ids[req]+histories[req][:3]))
 assert all(x==expected_input for x in all_inputs)
 assert all(x['replays']>=3 and x['captured']>0 and x['mode']=='FULL_DECODE_ONLY' for x in summary['graph'])
 return {'status':'passed' if all(x['passed'] for x in values) else 'failed','comparisons':values,'input_rows':len(expected_input)}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timeout',type=int,default=1800);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.timeout<=0:p.error('--timeout must be positive')
 base=a.output.parent/'forced_models';base.mkdir(parents=True,exist_ok=True);cases=[]
 for model,tp,batch in CASES:
  name=f'{model}_tp{tp}_b{batch}';directory=base/name;directory.mkdir(exist_ok=True)
  cmd=[sys.executable,'benchmarks/step105_forced_probe.py','--model',model,'--tp',str(tp),'--batch',str(batch),'--output',str(directory)]
  stage=run(cmd,directory/'run.log',a.timeout)
  item={'case':name,'stage':stage,'status':'failed'}
  if not stage['exit_code']:
   try:
    result=compare(model,tp,batch,directory);(directory/'compare.json').write_text(json.dumps(result,indent=2)+'\n')
    item.update(status=result['status'],comparisons=len(result['comparisons']),input_rows=result['input_rows'])
   except Exception as exc:
    import traceback
    item['error']=str(exc);item['traceback']=traceback.format_exc()
  cases.append(item);print(json.dumps(item),flush=True)
  a.output.write_text(json.dumps({'status':'running','cases':cases},indent=2)+'\n')
 result={'status':'passed' if all(x['status']=='passed' for x in cases) else 'failed','cases':cases}
 a.output.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
