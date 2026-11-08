"""Compare 101 8B fixed input against pre-registered frozen 090 references."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--probe-dir',type=Path);a=p.parse_args()
 contract=ROOT/'benchmarks/results/step090/numerics_contract.json';digest=hashlib.sha256(contract.read_bytes()).hexdigest()
 assert digest=='29bcbf64f1212cf100845387ff46ddcd43981b3d9fd4487c044a455ec28d0e7c'
 data=json.loads(contract.read_text());limits=data['models']['8b']['thresholds'];results=[]
 for rank in range(a.tp):
  m=torch.load((a.probe_dir or ROOT/f'benchmarks/results/step101/8b_tp{a.tp}')/f'step0_rank{rank}.pt',map_location='cpu',weights_only=True)
  v=torch.load(ROOT/f'benchmarks/results/step090/reference/8b_tp{a.tp}_kv/iter0_rank{rank}.pt',map_location='cpu',weights_only=True)
  assert torch.equal(m['input_ids'],v['input_ids']) and torch.equal(m['positions'],v['positions'])
  for mini,ref,threshold in (('layer0_qkv','qkv','qkv'),('layer0_attention','attention','attention'),('hidden','hidden','hidden'),('logits','logits','raw_logits')):
   x=m[mini].float();y=v[ref].float();assert x.shape==y.shape
   diff=(x-y).abs();index=int(diff.argmax());error=float(diff.flatten()[index]);relative=float((x-y).norm()/y.norm());lim=limits[threshold]
   top=None
   if mini=='logits':
    av,ai=x[0].topk(2);bv,bi=y[0].topk(2)
    top={'actual_ids':ai.tolist(),'reference_ids':bi.tolist(),'actual_values':av.tolist(),'reference_values':bv.tolist(),'actual_margin':float(av[0]-av[1]),'reference_margin':float(bv[0]-bv[1])}
   passed=error<=lim['max_abs'] and relative<=lim['relative_l2']
   results.append({'rank':rank,'object':threshold,'max_abs':error,'relative_l2':relative,'max_error_flat_index':index,'actual':float(x.flatten()[index]),'reference':float(y.flatten()[index]),'threshold':lim,'top':top,'passed':passed})
 result={'status':'passed' if all(x['passed'] for x in results) else 'failed','tp':a.tp,'contract_sha256':digest,'results':results};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'tp':a.tp,'status':result['status']}));return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
