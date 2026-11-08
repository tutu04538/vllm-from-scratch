"""Compare explicit same-token replay, including all rank input and KV mappings."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.step103_compare import summary_metric
from benchmarks.step103_workload import prompts


def logical_id(req):return int(req.split('-',1)[0])


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 matrix=json.loads(a.input.read_text());assert matrix['status']=='passed'
 contract_path=ROOT/'benchmarks/results/step103/numerics_contract.json';contract=json.loads(contract_path.read_text());assert contract['status']=='registered'
 out=[]
 for model,tps in [('8b',[1,2,4]),('14b',[2,4])]:
  for tp in tps:
   for batch in (1,4):
    pair={}
    for engine in ('frozen','mini'):
     rows=[x for x in matrix['cases'] if x['engine']==engine and x['model']==model and x['tp']==tp and x['batch']==batch]
     assert len(rows)==1 and rows[0]['exit_code']==0
     s=json.loads(Path(rows[0]['summary']).read_text());assert s['forced'] and all(len(z['tokens'])==4 for z in s['outputs'])
     rank_records=[];rank_inputs=[]
     for rank in range(tp):
      records=torch.load(s['ranks'][rank]['path'],map_location='cpu',weights_only=True)
      mapped={(logical_id(r['request_id']),r['sequence_length']-len(prompts(model,batch)[logical_id(r['request_id'])])):r for r in records}
      assert len(mapped)==batch*4
      rank_records.append(mapped)
      trace=torch.load(s['forced_ranks'][rank]['path'],map_location='cpu',weights_only=True)
      normalized=[]
      for e in trace:
       req=logical_id(e['request_id']);assert len(e['input_ids'])==len(e['positions'])==len(e['slot_mapping'])
       for token,pos,slot in zip(e['input_ids'],e['positions'],e['slot_mapping']):
        assert slot==e['block_table'][pos//16]*16+pos%16
        normalized.append((req,pos,token,slot))
      rank_inputs.append(normalized)
     assert all(trace==rank_inputs[0] for trace in rank_inputs)
     pair[engine]=(s,rank_records,[(r,pos,tok) for r,pos,tok,_ in rank_inputs[0]])
    frozen,mini=pair['frozen'],pair['mini']
    assert [x['tokens'] for x in frozen[0]['outputs']]==[x['tokens'] for x in mini[0]['outputs']]
    assert sorted(frozen[2])==sorted(mini[2])
    comparisons=[]
    for rank in range(tp):
     for req in range(batch):
      for step in range(4):
       f=frozen[1][rank][(req,step)];m=mini[1][rank][(req,step)]
       assert f['position']==m['position']==len(prompts(model,batch)[req])-1+step
       for obj in ('hidden','raw_logits','kv'):
        field='logits' if obj=='raw_logits' else obj;metrics=summary_metric(m[field],f[field])
        threshold=contract['thresholds'][f'{model}_b{batch}_step{step}_{obj}']
        passed=metrics['max_abs']<=threshold['max_abs'] and metrics['relative_l2']<=threshold['relative_l2']
        comparisons.append({'rank':rank,'request':req,'step':step,'object':obj,**metrics,'passed':passed})
    status='passed' if all(z['passed'] for z in comparisons) else 'failed'
    out.append({'model':model,'tp':tp,'batch':batch,'status':status,'comparisons':comparisons,
                'input_rows':len(frozen[2]),'physical_slot_policy':'per-rank table arithmetic; cross-engine block IDs need not match'})
    print(json.dumps({'model':model,'tp':tp,'batch':batch,'status':status,'comparisons':len(comparisons)}),flush=True)
 result={'status':'passed' if all(x['status']=='passed' for x in out) else 'failed','cases':out,
         'contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest()}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
