"""固定首分歧历史下比较完整 logits；只报差异，不自动调整门槛。"""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mini', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a=parser.parse_args()
    ref=json.loads(a.reference.read_text());other=json.loads(a.mini.read_text())
    if ref['case']!=other['case'] or ref['prompt_ids']!=other['prompt_ids']:
        raise ValueError('case or input mismatch')
    results=[]
    for request in ref['tokens']:
        left,right=ref['tokens'][request],other['tokens'][request]
        position=next((i for i,(x,y) in enumerate(zip(left,right)) if x!=y),None)
        if position is None: continue
        if left[:position]!=right[:position]: raise ValueError('histories differ before first divergence')
        per_rank=[]
        for rank in range(len(ref['ranks'])):
            def get(data):
                matches=[event for event in data['ranks'][rank]['events']
                         if event['request']==request and event['step']==position]
                if len(matches)!=1 or matches[0]['full_logits'] is None:
                    raise ValueError(f'missing full logits: {request}/{position}/{rank}')
                return matches[0]
            x,y=get(ref),get(other)
            lx=torch.tensor(x['full_logits']);ly=torch.tensor(y['full_logits'])
            if lx.shape!=ly.shape:raise ValueError('logits shape mismatch')
            diff=(lx-ly).abs();idx=int(diff.argmax())
            per_rank.append({'rank':rank,'max_abs':float(diff[idx]),
                'relative_l2':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(lx).clamp_min(1e-12)),
                'max_index':idx,'reference_value':float(lx[idx]),'mini_value':float(ly[idx]),
                'reference_top_ids':x['top_ids'],'reference_top_values':x['top_values'],
                'mini_top_ids':y['top_ids'],'mini_top_values':y['top_values']})
        results.append({'request':request,'first_different_index':position,
                        'reference_token':left[position],'mini_token':right[position],
                        'same_request_history':True,'ranks':per_rank})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'case':ref['case'],'first_differences':results},indent=2)+'\n')
    print(json.dumps({'requests':len(results),'sample':results[:1]}))

if __name__=='__main__':main()
