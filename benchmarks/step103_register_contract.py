"""Pre-register 103 workload tolerances solely from repeated/cross-TP frozen runs."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmarks/results/step103'


def read_case(model,tp,batch,iteration):
 path=OUT/f'reference_{model}_tp{tp}_b{batch}'
 summary=json.loads((path/'summary.json').read_text())
 outputs=summary['runs'][iteration]['outputs']
 mapped=[]
 for rank in range(tp):
  records=torch.load(path/f'run{iteration}_rank{rank}.pt',map_location='cpu',weights_only=True)
  by={}
  for i,output in enumerate(outputs):
   prefix=output['request_id']+'-'
   matches=[r for r in records if r['request_id']==output['request_id'] or r['request_id'].startswith(prefix)]
   assert len(matches)==4,(model,tp,batch,iteration,rank,output['request_id'],len(matches))
   for step,item in enumerate(matches):by[(i,step)]=item
  mapped.append(by)
 return summary,outputs,mapped


def metric(x,y):
 x=x.float();y=y.float();assert x.shape==y.shape,(x.shape,y.shape)
 if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('non-finite reference data')
 delta=x-y;denom=float(y.norm());norm=float(delta.norm())
 return {'max_abs':float(delta.abs().max()),'relative_l2':norm/denom if denom else (0. if norm==0 else float('inf'))}


def main():
 if list(OUT.glob('mini_*')):raise RuntimeError('cannot register after mini data exist')
 old=ROOT/'benchmarks/results/step090/numerics_contract.json'
 oldsha=hashlib.sha256(old.read_bytes()).hexdigest()
 assert oldsha=='29bcbf64f1212cf100845387ff46ddcd43981b3d9fd4487c044a455ec28d0e7c'
 registered={};details=[];mismatches=[]
 for model,tps in [('8b',[1,2,4]),('14b',[2,4])]:
  for batch in (1,4):
   cases={(tp,run):read_case(model,tp,batch,run) for tp in tps for run in (0,1)}
   base=cases[(tps[0],0)]
   for tp in tps:
    for run in (0,1):
     _,outputs,ranks=cases[(tp,run)]
     for req in range(batch):
      for step in range(4):
       if outputs[req]['tokens'][:step]!=base[1][req]['tokens'][:step]:
        mismatches.append({'model':model,'batch':batch,'tp':tp,'run':run,'request':req,'step':step});continue
       key=(req,step)
       obs=[torch.cat([r[key]['kv'] for r in ranks],dim=2),ranks[0][key]['hidden'],ranks[0][key]['logits']]
       ref=[torch.cat([r[key]['kv'] for r in base[2]],dim=2),base[2][0][key]['hidden'],base[2][0][key]['logits']]
       assert all(r[key]['position']==base[2][0][key]['position'] for r in ranks)
       for obj,x,y in zip(('kv','hidden','raw_logits'),obs,ref):
        m=metric(x,y); entry={'model':model,'batch':batch,'tp':tp,'run':run,'request':req,'step':step,'object':obj,**m};details.append(entry)
        k=f'{model}_b{batch}_step{step}_{obj}'
        if k not in registered:registered[k]={'observed_max_abs':0.,'observed_relative_l2':0.,'comparisons':0}
        registered[k]['observed_max_abs']=max(registered[k]['observed_max_abs'],m['max_abs'])
        registered[k]['observed_relative_l2']=max(registered[k]['observed_relative_l2'],m['relative_l2'])
        registered[k]['comparisons']+=1
 for key,value in registered.items():
  value['max_abs']=max(1e-5,value['observed_max_abs']*2)
  value['relative_l2']=max(1e-5,value['observed_relative_l2']*2)
 expected=2*2*4*3
 assert len(registered)==expected,(len(registered),expected)
 manifest={}
 for model,tps in [('8b',[1,2,4]),('14b',[2,4])]:
  for batch in (1,4):
   for tp in tps:
    directory=OUT/f'reference_{model}_tp{tp}_b{batch}'
    for f in [directory/'summary.json',*[directory/f'run{run}_rank{rank}.pt' for run in (0,1) for rank in range(tp)]]:
     manifest[str(f.relative_to(OUT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 out={'status':'registered' if not mismatches else 'registered_with_reference_history_mismatches',
      'definition':'2x max frozen repeat/cross-TP error; floor 1e-5, pre-registered before mini results',
      'old_contract_sha256':oldsha,'thresholds':registered,'reference_history_mismatches':mismatches,
      'reference_comparisons':details,'reference_sha256':manifest,
      'reference_harness_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'benchmarks/step103_frozen_probe.py',ROOT/'benchmarks/step103_hooks.py',ROOT/'benchmarks/step103_workload.py']}}
 path=OUT/'numerics_contract.json';path.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'thresholds':len(registered),'history_mismatches':len(mismatches),
                   'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
 return 0 if not mismatches else 1
if __name__=='__main__':sys.exit(main())
