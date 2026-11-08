"""Validate isolated memory probes and constrained random rank agreement."""
import argparse,json,sys
from pathlib import Path
import torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 matrix=json.loads(a.input.read_text());assert matrix['status']=='passed'
 results=[]
 for case in matrix['cases']:
  summary=json.loads(Path(case['summary']).read_text());tp=case['tp']
  assert len(summary['before'])==len(summary['after'])==len(summary['ranks'])==tp
  assert sorted(row['rank'] for row in summary['after'])==list(range(tp))
  assert len({row['uuid'] for row in summary['after']})==tp
  for before,after in zip(summary['before'],summary['after']):
   assert before['parameter_bytes']>0 and before['kv_tensor_bytes']>0
   assert before['parameter_bytes']==after['parameter_bytes'] and before['kv_tensor_bytes']==after['kv_tensor_bytes']
   assert after['peak_allocated_bytes']>=after['allocated_bytes']>=after['parameter_bytes']
   assert after['peak_reserved_bytes']>=after['reserved_bytes']>=after['allocated_bytes']
   assert after['known_workspace_bytes']>=0
  checked=0
  if case['random']:
   records=[torch.load(row['path'],map_location='cpu',weights_only=True) for row in summary['ranks']]
   by_rank=[{(r['request_id'],r['position']):r for r in rank} for rank in records]
   assert all(set(rank)==set(by_rank[0]) for rank in by_rank)
   assert len(by_rank[0])==case['batch']*4
   for key in by_rank[0]:
    rows=[rank[key] for rank in by_rank];token=rows[0]['sampled']
    assert all(r['sampled']==token for r in rows)
    for row in rows:
     logits=row['logits'].float();assert logits[token]>=torch.topk(logits,5).values[-1]
     checked+=1
  results.append({'engine':case['engine'],'model':case['model'],'tp':tp,'batch':case['batch'],
                  'random':case['random'],'rank_sample_checks':checked,'status':'passed'})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'status':'passed','cases':results},indent=2)+'\n')
 print(json.dumps({'status':'passed','cases':len(results),'rank_sample_checks':sum(x['rank_sample_checks'] for x in results)}))
if __name__=='__main__':main()
