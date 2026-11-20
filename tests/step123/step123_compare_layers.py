"""Compare captured embedding, decoder layer and final norm tensors."""
import argparse
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--mini', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
a = p.parse_args()
mini = torch.load(a.mini / 'layers_rank0.pt', map_location='cpu', weights_only=True)
reference = torch.load(a.reference / 'layers_rank0.pt', map_location='cpu', weights_only=True)
assert mini.keys() == reference.keys()
errors = []
for key, outputs in mini.items():
    assert len(outputs) == len(reference[key])
    for x, y in zip(outputs, reference[key]):
        assert (x is None) == (y is None)
        if x is not None:
            assert x.shape == y.shape
            errors.append(float((x.float() - y.float()).abs().max()))
print({'keys': len(mini), 'tensors': len(errors), 'max_abs': max(errors)})
