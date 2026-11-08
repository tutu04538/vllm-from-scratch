"""Compare same-history 103 traces against pre-registered frozen-only limits."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmarks/results/step103'
sys.path.insert(0,str(ROOT))
from benchmarks.step103_register_contract import read_case,metric


def read_mini(path,tp,batch):
 summary=json.loads((path/'summary.json').read_text());outputs=summary['runs'][0]['outputs'];mapped=[]
 for rank in range(tp):
  records=torch.load(path/f'run0_rank{rank}.pt',map_location='cpu',weights_only=True);by={}
  for i,out in enumerate(outputs):
   matches=[r for r in records if r['request_id']==out['request_id']]
   assert len(matches)==4,(i,rank,len(matches))
   for step,item in enumerate(matches):by[(i,step)]=item
  mapped.append(by)
 return summary,outputs,mapped


def summary_metric(x,y):
 x=x.float();y=y.float();assert x.shape==y.shape
 diff=(x-y).abs();index=int(diff.argmax());m=metric(x,y)
 return {**m,'max_error_flat_index':index,'actual':float(x.flatten()[index]),'reference':float(y.flatten()[index])}


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['8b','14b'],required=True);p.add_argument('--tp',type=int,required=True)
 p.add_argument('--batch',type=int,choices=[1,4],required=True);p.add_argument('--mini-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 contract_path=OUT/'numerics_contract.json';contract=json.loads(contract_path.read_text());assert contract['status']=='registered'
 refdir=OUT/f'reference_{a.model}_tp{a.tp}_b{a.batch}'
 for f in [refdir/'summary.json',*[refdir/f'run{run}_rank{rank}.pt' for run in (0,1) for rank in range(a.tp)]]:
  assert hashlib.sha256(f.read_bytes()).hexdigest()==contract['reference_sha256'][str(f.relative_to(OUT))],f
 reference,refout,refranks=read_case(a.model,a.tp,a.batch,0)
 mini,miniout,miniranks=read_mini(a.mini_dir,a.tp,a.batch)
 assert mini['prompts']==reference['prompts']
 cases=[];history_mismatch=[];full_generation=[]
 for req in range(a.batch):
  a_tokens=miniout[req]['tokens'];b_tokens=refout[req]['tokens'];assert len(a_tokens)==len(b_tokens)==32
  first=next((i for i,(x,y) in enumerate(zip(a_tokens,b_tokens)) if x!=y),None)
  full_generation.append({'request':req,'exact':first is None,'first_divergence':first,'mini_tokens':a_tokens,'reference_tokens':b_tokens})
  for step in range(4):
   key=(req,step)
   if a_tokens[:step]!=b_tokens[:step]:
    history_mismatch.append({'request':req,'step':step,'first_divergence':first});continue
   for rank in range(a.tp):
    m=miniranks[rank][key];r=refranks[rank][key]
    assert m['position']==r['position'] and m['sequence_length']==r['sequence_length']
    for obj in ('hidden','raw_logits','kv'):
     field='logits' if obj=='raw_logits' else obj
     values=summary_metric(m[field],r[field]);limits=contract['thresholds'][f'{a.model}_b{a.batch}_step{step}_{obj}']
     passed=values['max_abs']<=limits['max_abs'] and values['relative_l2']<=limits['relative_l2']
     top=None
     if obj=='raw_logits':
      av,ai=m[field].float().topk(2);bv,bi=r[field].float().topk(2)
      top={'actual_ids':ai.tolist(),'reference_ids':bi.tolist(),'actual_values':av.tolist(),
           'reference_values':bv.tolist(),'actual_margin':float(av[0]-av[1]),'reference_margin':float(bv[0]-bv[1])}
     cases.append({'request':req,'step':step,'rank':rank,'object':obj,**values,'limits':{'max_abs':limits['max_abs'],'relative_l2':limits['relative_l2']},'top':top,'passed':passed})
 expected=a.batch*4*a.tp*3
 result={'status':'passed' if len(cases)==expected and all(c['passed'] for c in cases) and all(x['exact'] for x in full_generation) else 'failed',
   'model':a.model,'tp':a.tp,'batch':a.batch,'contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),
   'comparisons':cases,'expected_comparisons':expected,'history_mismatches':history_mismatch,'generation':full_generation,
   'mini_rank_memory':mini['initial_ranks'],'reference_rank_memory':reference['initial_ranks'],
   'finish_steps':[x['finish_step'] for x in miniout]}
 if a.batch==4:assert len(set(result['finish_steps']))>1,'B4 requests did not finish in different steps'
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'model':a.model,'tp':a.tp,'batch':a.batch,'compared':len(cases),'expected':expected,
                   'numeric_failures':sum(not c['passed'] for c in cases),'generation_divergences':sum(not x['exact'] for x in full_generation)}))
 return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
