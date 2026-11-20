"""Check actual AWQ TP lifecycle traces and compare identical TP1/TP2 runs."""
import argparse
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
a = p.parse_args()

for mode in ('prefix', 'preempt', 'cancel_before', 'cancel_after'):
    runs = [torch.load(a.directory / f'step125-lifecycle-{mode}-tp{tp}.pt',
                       map_location='cpu', weights_only=True) for tp in (1, 2)]
    for run in runs:
        for row in run['rounds']:
            ranks = row['ranks']
            assert all(r['requests'] == ranks[0]['requests'] and
                       r['trace'] == ranks[0]['trace'] and
                       r['trace_count'] == ranks[0]['trace_count'] for r in ranks)
        assert not run['rounds'][-1]['ranks'][0]['requests']
        assert not any(run['rounds'][-1]['refcounts'])
    assert len(runs[0]['rounds']) == len(runs[1]['rounds'])
    for single, multi in zip(runs[0]['rounds'], runs[1]['rounds']):
        assert single['outputs'] == multi['outputs']
        assert single['scheduler'] == multi['scheduler']
        assert single['trace'] == multi['trace']
    if mode == 'prefix':
        assert any(r['trace'] and r['trace']['hits'].get('b', 0) >= 4
                   for r in runs[1]['rounds'])
    if mode == 'preempt':
        assert runs[1]['rounds'][-1]['preemptions'] > 0
    print(mode, 'passed', len(runs[1]['rounds']), 'rounds')

failure = torch.load(a.directory / 'step125-lifecycle-fail-tp2.pt',
                     map_location='cpu', weights_only=True)
assert failure['before'] == failure['after'] and failure['failure']
print('rank failure passed')
