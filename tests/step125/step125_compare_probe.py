"""Compare the same real-model history and recorded ranks with frozen vLLM."""
import argparse
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--mini', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--tp', type=int, choices=(1, 2, 4), required=True)
a = p.parse_args()
mini = json.loads((a.mini / 'summary.json').read_text())
reference = json.loads((a.reference / 'summary.json').read_text())
assert mini['prompts'] == reference['prompts']
mini_tokens = mini['runs'][0]['outputs'][0]['tokens']
ref_tokens = reference['runs'][0]['outputs'][0]['tokens']
assert len(mini_tokens) == len(ref_tokens) == 128
first = next((i for i, (x, y) in enumerate(zip(mini_tokens, ref_tokens)) if x != y), None)
print('128 token first difference:', first)
for rank in range(a.tp):
    m = torch.load(a.mini / f'run0_rank{rank}.pt', map_location='cpu', weights_only=True)
    v = torch.load(a.reference / f'run0_rank{rank}.pt', map_location='cpu', weights_only=True)
    assert len(m) == len(v) == 4
    for i, (x, y) in enumerate(zip(m, v)):
        assert x['position'] == y['position']
        assert mini_tokens[:i] == ref_tokens[:i], 'different history prevents numerical comparison'
        print('rank', rank, 'step', i,
              {key: float((x[key].float() - y[key].float()).abs().max())
               for key in ('hidden', 'logits', 'kv')})
