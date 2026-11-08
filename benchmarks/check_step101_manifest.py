"""Audit every local parameter source interval against the checkpoint index and shapes."""
import argparse,json,sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.check_step090_models import resolve_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['8b','14b'],required=True);p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--probe-dir',type=Path);a=p.parse_args()
 path,_=resolve_snapshot(a.model);index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map'];results=[]
 with ExitStack() as stack:
  handles={file:stack.enter_context(safe_open(path/file,framework='pt',device='cpu')) for file in set(index.values())}
  for rank in range(a.tp):
   manifest=json.loads(((a.probe_dir or ROOT/f'benchmarks/results/step101/{a.model}_tp{a.tp}')/f'manifest_rank{rank}.json').read_text());names=[]
   for item in manifest['parameters']:
    shapes=[]
    for src in item['sources']:
     name=src['checkpoint'];names.append(name);assert name in index,name
     shape=list(handles[index[name]].get_slice(name).get_shape());axis=src['axis']
     if axis is not None:
      assert 0<=src['start']<=src['end']<=shape[axis],(name,src,shape)
      shape[axis]=src['end']-src['start']
     shapes.append(shape)
    expected=shapes[0]
    if len(shapes)>1:
     assert all(s[1:]==expected[1:] for s in shapes);expected=[sum(s[0] for s in shapes),*expected[1:]]
    actual=item['shape']
    if item['name'].endswith(('embed_tokens.weight','lm_head.weight')):
     assert actual[1:]==expected[1:] and actual[0]>=expected[0]
    else:assert actual==expected,(item['name'],actual,expected)
   counts=Counter(names);assert set(counts)==set(index),(set(index)-set(counts),set(counts)-set(index));assert all(n==1 for n in counts.values())
   results.append({'rank':rank,'parameters':len(manifest['parameters']),'source_tensors':len(names),'duplicate_sources':0,'missing_sources':0,'parameter_bytes':manifest['parameter_bytes'],'peak_allocated':manifest['peak_allocated'],'peak_reserved':manifest['peak_reserved'],'uuid':manifest['uuid']})
 result={'model':a.model,'tp':a.tp,'status':'passed','results':results};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'model':a.model,'tp':a.tp,'status':'passed'}))
if __name__=='__main__':main()
