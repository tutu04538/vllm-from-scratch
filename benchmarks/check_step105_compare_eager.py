"""Compare 105 real Graph directly against 104's eager real-model rerun."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.check_step105_models import CASES
from benchmarks.step103_compare import summary_metric
from benchmarks.step103_workload import prompts


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 contract=json.loads((ROOT/'benchmarks/results/step103/numerics_contract.json').read_text())['thresholds'];cases=[]
 for model,tp,batch in CASES:
  name=f'{model}_tp{tp}_b{batch}';e=ROOT/f'benchmarks/results/step104/real_model_regression/{name}';g=ROOT/f'benchmarks/results/step105/graph_models/{name}'
  eager=json.loads((e/'summary.json').read_text());graph=json.loads((g/'summary.json').read_text())
  assert [x['tokens'] for x in eager['runs'][0]['outputs']]==[x['tokens'] for x in graph['runs'][0]['outputs']]
  ids=prompts(model,batch);values=[]
  for rank in range(tp):
   def keyed(path):
    rows=torch.load(path,map_location='cpu',weights_only=True);return {(int(x['request_id'].split('-',1)[0]),x['sequence_length']-len(ids[int(x['request_id'].split('-',1)[0])])):x for x in rows}
   left=keyed(e/f'run0_rank{rank}.pt');right=keyed(g/f'run0_rank{rank}.pt');assert left.keys()==right.keys() and len(left)==batch*4
   for key in left:
    req,step=key;x=left[key];y=right[key];assert x['position']==y['position']
    for field,obj in [('hidden','hidden'),('logits','raw_logits'),('kv','kv')]:
     metrics=summary_metric(y[field],x[field]);limit=contract[f'{model}_b{batch}_step{step}_{obj}']
     values.append({'rank':rank,'request':req,'step':step,'object':obj,**metrics,'passed':metrics['max_abs']<=limit['max_abs'] and metrics['relative_l2']<=limit['relative_l2']})
  cases.append({'case':name,'status':'passed' if all(x['passed'] for x in values) else 'failed','comparisons':values})
  print(json.dumps({'case':name,'status':cases[-1]['status'],'comparisons':len(values)}),flush=True)
 result={'status':'passed' if all(x['status']=='passed' for x in cases) else 'failed','cases':cases}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
