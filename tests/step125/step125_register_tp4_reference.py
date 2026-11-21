"""Freeze upstream-only TP4 references before running mini TP4."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
a = p.parse_args()
contract = {'scope': 'TP4 AWQ V2 BF16 eager TRITON_ATTN',
            'thresholds': {'logits_rtol': 1e-2, 'logits_atol': 5e-2,
                           'logits_mean_abs_max': 1e-2},
            'references': {}, 'repeats': {}}
for size in ('8b', '14b'):
    for batch in (1, 4):
        case = f'{size}_b{batch}'
        directory = a.directory / f'ref_{case}'
        summary = json.loads((directory / 'summary.json').read_text())
        assert len(summary['runs']) == 2
        files = [directory / 'summary.json']
        for rank in range(4):
            files += [directory / f'run{i}_rank{rank}.pt' for i in range(2)]
            records = [torch.load(directory / f'run{i}_rank{rank}.pt',
                                  map_location='cpu', weights_only=True) for i in range(2)]
            for req in range(batch):
                actual = []
                for run in range(2):
                    req_id = summary['runs'][run]['outputs'][req]['request_id']
                    actual.append([r for r in records[run]
                                   if r['request_id'].split('-')[0] == req_id])
                assert len(actual[0]) == len(actual[1]) and actual[0]
                for step, (left, right) in enumerate(zip(*actual)):
                    assert left['position'] == right['position']
                    for field in ('hidden', 'logits', 'kv'):
                        x, y = left[field].float(), right[field].float()
                        assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
                        delta = (x-y).abs()
                        contract['repeats'][f'{case}/rank{rank}/req{req}/step{step}/{field}'] = {
                            'max_abs': float(delta.max()), 'mean_abs': float(delta.mean())}
        for file in files:
            contract['references'][str(file.relative_to(a.directory))] = hashlib.sha256(file.read_bytes()).hexdigest()
path = a.directory / 'reference_contract.json'
assert not path.exists(), 'reference contract already frozen'
path.write_text(json.dumps(contract, indent=2, sort_keys=True) + '\n')
print('contract', path, 'sha256', hashlib.sha256(path.read_bytes()).hexdigest())
print('repeat_max_abs', max(x['max_abs'] for x in contract['repeats'].values()))
