"""Check real draft mapped logits and normalized q in every TP rank."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--temperature',type=float,default=.8)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.temperature<=0:p.error('temperature must be positive')
    summary=json.loads((a.trace/'summary.json').read_text())
    counts=[]
    for rank in range(summary['tp']):
        events=torch.load(a.trace/f'state_rank{rank}.pt',weights_only=True)
        count=0;nonintersection=None
        for event in events:
            if event['kind']!='propose':continue
            logits=event['draft_logits']
            if logits is None:raise AssertionError('missing probabilistic draft logits')
            if logits.shape[-1] != 151936:raise AssertionError(f'unexpected target vocabulary width: {logits.shape}')
            if torch.isnan(logits).any() or torch.isposinf(logits).any():raise AssertionError('invalid mapped logits')
            # CPU FP64 accumulation avoids mistaking reduction error across
            # 151936 entries for missing probability mass.
            q=torch.softmax(logits.double()/a.temperature,dim=-1)
            torch.testing.assert_close(q.sum(dim=-1),torch.ones_like(q.sum(dim=-1)),rtol=1e-12,atol=1e-12)
            absent=torch.isneginf(logits)
            if not absent.any():raise AssertionError('missing nonintersection mask')
            if torch.count_nonzero(q[absent]):raise AssertionError('nonintersection has nonzero mass')
            nonintersection=int(absent.sum());count+=1
        if not count:raise AssertionError('no proposal events')
        counts.append({'rank':rank,'proposals':count,'masked_elements_last':nonintersection})
    result={'status':'passed','temperature':a.temperature,'ranks':counts}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
