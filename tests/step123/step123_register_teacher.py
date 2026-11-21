"""Freeze new multi-prompt AWQ teacher-forcing references before mini runs."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
a = p.parse_args()
contract = {'case': 'teacher_variant_1_batch4', 'limits': {'rtol': .01, 'atol': .05,
            'mean_abs_max': .01}, 'reference_hashes': {}, 'repeats': {}}
for size in ('8b', '14b'):
    for tp in (1, 2):
        case=f'{size}_tp{tp}'
        directory=a.directory/f'ref_{case}'
        summary=json.loads((directory/'summary.json').read_text())
        assert len(summary['runs'])==2 and len(summary['prompts'])==4
        files=[directory/'summary.json']
        for rank in range(tp):
            files += [directory/f'run{i}_rank{rank}.pt' for i in range(2)]
            records=[torch.load(directory/f'run{i}_rank{rank}.pt',map_location='cpu',weights_only=True) for i in range(2)]
            for request in range(4):
                items=[]
                for run in range(2):
                    rid=summary['runs'][run]['outputs'][request]['request_id']
                    matches=[item for item in records[run] if item['request_id'].split('-')[0]==rid]
                    assert len(matches)==1
                    items.append(matches[0])
                assert items[0]['position']==items[1]['position']
                for field in ('hidden','logits','kv'):
                    x,y=items[0][field].float(),items[1][field].float()
                    assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
                    d=(x-y).abs()
                    contract['repeats'][f'{case}/rank{rank}/req{request}/{field}']={'max_abs':float(d.max()),'mean_abs':float(d.mean())}
        for f in files:
            contract['reference_hashes'][str(f.relative_to(a.directory))]=hashlib.sha256(f.read_bytes()).hexdigest()
path=a.directory/'reference_contract.json'
assert not path.exists()
path.write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
print('contract_sha256',hashlib.sha256(path.read_bytes()).hexdigest())
print('repeat_max_abs',max(x['max_abs'] for x in contract['repeats'].values()))
