"""Report fixed-history logits errors; enforce the initial numeric limits."""
import argparse
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--mini', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--tp', type=int, choices=(1, 2, 4), default=1)
a = p.parse_args()
mini = json.loads((a.mini / 'summary.json').read_text())
reference = json.loads((a.reference / 'summary.json').read_text())
assert mini['prompts'] == reference['prompts']
failed = False
for rank in range(a.tp):
    m = torch.load(a.mini / f'run0_rank{rank}.pt', map_location='cpu', weights_only=True)
    r = torch.load(a.reference / f'run0_rank{rank}.pt', map_location='cpu', weights_only=True)
    for i in range(len(mini['prompts'])):
        x = next(item for item in m if item['request_id'] == str(i))
        y = next(item for item in r if item['request_id'].split('-')[0] == str(i))
        assert x['position'] == y['position']
        u, v = x['logits'].float(), y['logits'].float()
        assert u.shape == v.shape and torch.isfinite(u).all() and torch.isfinite(v).all()
        d = (u-v).abs()
        close = torch.isclose(u, v, rtol=1e-2, atol=5e-2)
        passed = bool(close.all()) and float(d.mean()) <= 1e-2
        failed |= not passed
        print(json.dumps({'rank': rank, 'request': i, 'max_abs': float(d.max()),
                          'mean_abs': float(d.mean()), 'p99_abs': float(d.quantile(.99)),
                          'outside_tolerance': int((~close).sum()),
                          'top1_mini': int(u.argmax()), 'top1_reference': int(v.argmax()),
                          'passed': passed}))
raise SystemExit(1 if failed else 0)
