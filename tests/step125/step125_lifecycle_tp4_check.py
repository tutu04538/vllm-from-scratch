"""Check real AWQ TP4 scheduler/worker lifecycle against recorded TP1."""
import argparse
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
p.add_argument('--single', type=Path, required=True)
a = p.parse_args()
for mode in ('prefix', 'preempt', 'cancel_before', 'cancel_after'):
    multi = torch.load(a.directory / f'step125-lifecycle-{mode}-tp4.pt', weights_only=True)
    single = torch.load(a.single / f'step125-lifecycle-{mode}-tp1.pt', weights_only=True)
    assert len(multi['rounds']) == len(single['rounds'])
    for left, right in zip(multi['rounds'], single['rounds']):
        assert len(left['ranks']) == 4
        base = left['ranks'][0]
        assert all(r['requests'] == base['requests'] and r['trace'] == base['trace']
                   and r['trace_count'] == base['trace_count'] for r in left['ranks'])
        assert left['outputs'] == right['outputs']
        assert left['scheduler'] == right['scheduler']
        assert left['trace'] == right['trace']
    assert not multi['rounds'][-1]['ranks'][0]['requests']
    assert not any(multi['rounds'][-1]['refcounts'])
    if mode == 'prefix':
        assert any(r['trace'] and r['trace']['hits'].get('b', 0) >= 4 for r in multi['rounds'])
    if mode == 'preempt':
        assert multi['rounds'][-1]['preemptions'] > 0
    print(mode, 'passed', len(multi['rounds']), 'rounds')
failure = torch.load(a.directory / 'step125-lifecycle-fail-tp4.pt', weights_only=True)
assert failure['before'] == failure['after'] and failure['failure']
print('rank failure passed')
