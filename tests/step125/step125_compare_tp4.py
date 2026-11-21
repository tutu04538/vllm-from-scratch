"""Compare TP4 against the previously frozen reference and its fixed limits."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
contract = json.loads((a.directory / 'reference_contract.json').read_text())
for name, expected in contract['references'].items():
    assert hashlib.sha256((a.directory / name).read_bytes()).hexdigest() == expected, name
result = {'cases': [], 'contract_sha256': hashlib.sha256(
    (a.directory / 'reference_contract.json').read_bytes()).hexdigest()}
for size in ('8b', '14b'):
    for batch in (1, 4):
        case = f'{size}_b{batch}'
        mini_dir, ref_dir = a.directory / f'mini_{case}', a.directory / f'ref_{case}'
        mini = json.loads((mini_dir / 'summary.json').read_text())
        ref = json.loads((ref_dir / 'summary.json').read_text())
        assert mini['prompts'] == ref['prompts']
        for rank in range(4):
            m = torch.load(mini_dir / f'run0_rank{rank}.pt', map_location='cpu', weights_only=True)
            v = torch.load(ref_dir / f'run0_rank{rank}.pt', map_location='cpu', weights_only=True)
            for req in range(batch):
                mt = mini['runs'][0]['outputs'][req]['tokens']
                rt = ref['runs'][0]['outputs'][req]['tokens']
                mid = mini['runs'][0]['outputs'][req]['request_id']
                vid = ref['runs'][0]['outputs'][req]['request_id']
                mr = [x for x in m if x['request_id'] == mid]
                vr = [x for x in v if x['request_id'].split('-')[0] == vid]
                assert len(mr) == len(vr) and mr
                for step, (left, right) in enumerate(zip(mr, vr)):
                    assert mt[:step] == rt[:step] and left['position'] == right['position']
                    for field in ('hidden', 'logits', 'kv'):
                        x,y=left[field].float(),right[field].float()
                        assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
                        delta=(x-y).abs()
                        # Logits use the requirement's pre-existing default contract. Other
                        # fields are reported explicitly; exact equality is a stricter diagnostic.
                        if field == 'logits':
                            passed = bool(torch.isclose(x,y,rtol=1e-2,atol=5e-2).all()) and float(delta.mean()) <= 1e-2
                        else:
                            passed = bool(torch.equal(x,y))
                        result['cases'].append({'case':case,'rank':rank,'request':req,'step':step,
                            'field':field,'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'passed':passed})
                result.setdefault('generation', []).append({'case':case,'rank':rank,'request':req,
                    'exact':mt==rt,'first_difference':next((i for i,(x,y) in enumerate(zip(mt,rt)) if x!=y),None)})
result['status'] = 'passed' if all(x['passed'] for x in result['cases']) and all(x['exact'] for x in result['generation']) else 'failed'
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'comparisons':len(result['cases']),
 'numeric_failed':sum(not x['passed'] for x in result['cases']),
 'generation_failed':sum(not x['exact'] for x in result['generation'])}))
raise SystemExit(0 if result['status']=='passed' else 1)
