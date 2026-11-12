"""Register frozen-repeat thresholds and compare real EAGLE3 state traces."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f'tensor contract differs: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}')
    if torch.isnan(a).any() or torch.isnan(b).any():
        raise ValueError('NaN in compared tensor')
    if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
        raise ValueError('finite mask differs')
    if not torch.equal(torch.isneginf(a), torch.isneginf(b)) or not torch.equal(torch.isposinf(a), torch.isposinf(b)):
        raise ValueError('infinity mask differs')
    finite = torch.isfinite(a)
    if not finite.any():
        return {'max_abs':0., 'relative_l2':0., 'max_index':None}
    x = a[finite].float(); y = b[finite].float(); delta = (x-y).abs()
    index = int(delta.argmax()); full = torch.nonzero(finite.flatten(), as_tuple=False).flatten()[index].item()
    coords = list(torch.unravel_index(torch.tensor(full), a.shape))
    return {'max_abs':float(delta[index]), 'relative_l2':float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp_min(1e-30)),
            'max_index':[int(i) for i in coords], 'reference_value':float(x[index]), 'observed_value':float(y[index])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--other', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--register', action='store_true')
    a = p.parse_args()
    ref = json.loads((a.reference/'summary.json').read_text())
    other = json.loads((a.other/'summary.json').read_text())
    for key in ('model','tp','batch','k','mode','prompts'):
        if ref[key] != other[key]: raise ValueError(f'configuration mismatch {key}')
    if a.register and other['implementation'] != 'frozen': raise ValueError('repeat must be frozen reference')
    if ref['implementation'] != 'frozen': raise ValueError('reference must be frozen')
    if not a.register:
        contract = json.loads(a.contract.read_text())
        if contract['reference_digest'] != digest(a.reference/'summary.json'):
            raise ValueError('reference differs from pre-registered source')
    else:
        contract = {'reference_digest':digest(a.reference/'summary.json'), 'thresholds':{}}
    report = {'passed':True, 'first_diff':None, 'values':{}, 'top_margins':[], 'discrete_equal':ref['tokens']==other['tokens']}
    if not report['discrete_equal']:
        report['passed']=False
    for rank in range(ref['tp']):
        x=torch.load(a.reference/f'state_rank{rank}.pt',weights_only=True)
        y=torch.load(a.other/f'state_rank{rank}.pt',weights_only=True)
        if len(x)!=len(y):
            report['passed']=False
            if report['first_diff'] is None: report['first_diff']=f'rank{rank} event count {len(x)} != {len(y)}'
        for i,(left,right) in enumerate(zip(x,y)):
            if left.keys()!=right.keys() or left['kind']!=right['kind']:
                report['passed']=False
                if report['first_diff'] is None: report['first_diff']=f'rank{rank} event {i} mismatched fields'
                break
            for key in left:
                u,v=left[key],right[key]; label=f'rank{rank}/{i}/{key}'
                if isinstance(u,torch.Tensor):
                    if u.is_floating_point():
                        try:
                            result=measure(u,v)
                        except ValueError as exc:
                            report['passed']=False
                            if report['first_diff'] is None: report['first_diff']=f'{label}: {exc}'
                            continue
                        if a.register: contract['thresholds'][label]=max(1e-5,2*result['max_abs'])
                        else:
                            threshold=contract['thresholds'][label]
                            result['threshold']=threshold
                            if result['max_abs']>threshold:
                                report['passed']=False
                                if report['first_diff'] is None: report['first_diff']=label
                        report['values'][label]=result
                        if key=='logits' and u.numel():
                            vals,ids=u.float().topk(2,dim=-1)
                            report['top_margins'].append({'rank':rank,'event':i,'top1':ids[...,0].tolist(),
                                'top2':ids[...,1].tolist(),'margin':(vals[...,0]-vals[...,1]).tolist()})
                    elif not torch.equal(u,v):
                        report['passed']=False
                        if report['first_diff'] is None: report['first_diff']=label
                elif isinstance(u,list) and u and isinstance(u[0],torch.Tensor):
                    if len(u)!=len(v) or any(not torch.equal(l,r) for l,r in zip(u,v)):
                        report['passed']=False
                        if report['first_diff'] is None: report['first_diff']=label
                elif u!=v:
                    report['passed']=False
                    if report['first_diff'] is None: report['first_diff']=label
    if report['first_diff'] is None and not report['discrete_equal']:
        report['first_diff']='final tokens'
    if a.register:
        if not report['passed']: raise ValueError(f'reference repeats differ in discrete state: {report["first_diff"]}')
        a.contract.write_text(json.dumps(contract,indent=2)+'\n')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'], 'first_diff':report['first_diff'], 'compared':len(report['values'])}))
    return 0 if report['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
